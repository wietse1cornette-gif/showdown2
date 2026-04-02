#git --version (checken of het al aanwezig is en anders VS Code terug afsluiten)
#git clone https://github.com/wietse1cornette-gif/showdown2.git
#cd showdown2

import cv2
import numpy as np
import time
import winsound

# --- CONFIGURATIE ---
cap = cv2.VideoCapture(0)

def speel_buzzer():
    winsound.Beep(1000, 300)
    print("BEEP! Lijn overtreden!")

def vind_gele_maan(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    geel_laag = np.array([15, 100, 100])
    geel_hoog = np.array([35, 255, 255])
    masker = cv2.inRange(hsv, geel_laag, geel_hoog)
    masker = cv2.morphologyEx(masker, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    masker = cv2.dilate(masker, None, iterations=2)
    
    contouren, _ = cv2.findContours(masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contouren:
        return None
    
    grootste = max(contouren, key=cv2.contourArea)
    if cv2.contourArea(grootste) < 500:
        return None
    
    x, y, w, h = cv2.boundingRect(grootste)
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy, w // 2 + 10, h // 2 + 10)

prev_frame = None
laatste_beep_tijd = 0
BEEP_COOLDOWN = 0.5
vaste_positie = None

# --- AANPASBARE OFFSET (positief = boog verder naar beneden) ---
Y_OFFSET = 0  # Pas dit aan als de boog nog niet goed zit
X_OFFSET = 0  # Niet gebruikt, maar kan handig zijn voor horizontale aanpassingen

print("=" * 50)
print("Druk op 'K' om de gele maan te kalibreren.")
print("Druk op '+' of '-' om de boog omhoog/omlaag te schuiven.")
print("Druk op 'l' of 'r' om de boog horizontaal te schuiven (optioneel).")
print("Druk op 'Q' om te stoppen.")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    frame_delta = cv2.absdiff(prev_frame, gray)
    prev_frame = gray
    _, motion_mask = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    goal_zone_mask = np.zeros_like(motion_mask)

    if vaste_positie:
        cx, cy, rx, ry = vaste_positie
        # Y_OFFSET schuift de boog naar beneden op de maan
        draw_cy = cy + Y_OFFSET
        draw_cx = cx + X_OFFSET

        # 180-360 graden = bovenste boog (omgekeerde U)
        cv2.ellipse(frame, (draw_cx, draw_cy), (rx, ry), 0, 0, 180, (255, 0, 0), 2)
        cv2.ellipse(goal_zone_mask, (draw_cx, draw_cy), (rx, ry), 0, 0, 180, 255, 6)

        cv2.putText(frame, f"VAST: ({draw_cx},{draw_cy}) offset={Y_OFFSET}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Druk 'K' om te kalibreren!",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    motion_on_line = cv2.bitwise_and(motion_mask, goal_zone_mask)
    motion_pixels = np.sum(motion_on_line == 255)

    huidige_tijd = time.time()
    if vaste_positie and motion_pixels > 400 and (huidige_tijd - laatste_beep_tijd) > BEEP_COOLDOWN:
        cv2.putText(frame, "LINE VIOLATION!", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        speel_buzzer()
        laatste_beep_ijd = huidige_tijd
        print(f"Beweging op de lijn: {motion_pixels} pixels")

    cv2.imshow('1. Live Beeld', frame)
    cv2.imshow('2. Bewegings Filter', motion_on_line)

    toets = cv2.waitKey(1) & 0xFF

    if toets == ord('k'):
        positie = vind_gele_maan(frame)
        if positie:
            vaste_positie = positie
            print(f"✓ Gekalibreerd op: {vaste_positie}, offset={Y_OFFSET}")
        else:
            print("✗ Geen gele maan gevonden!")

    elif toets == ord('+'):
        Y_OFFSET += 5
        print(f"Boog naar beneden: offset={Y_OFFSET}")

    elif toets == ord('-'):
        Y_OFFSET -= 5
        print(f"Boog naar boven: offset={Y_OFFSET}")
    
    elif toets == ord('l'):
        X_OFFSET += 5
        print(f"Boog naar rechts: offset={X_OFFSET}")

    elif toets == ord('r'):
        X_OFFSET -= 5
        print(f"Boog naar links: offset={X_OFFSET}")

    elif toets == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()