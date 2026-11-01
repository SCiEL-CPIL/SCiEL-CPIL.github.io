import numpy as np

def construct():
    # 12 unit hexagons
    # Center
    centers = [(0.0, 0.0)]
    
    # Ring 1 (6 hexagons)
    r1 = np.sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
        
    # Ring 2 Mid (6 hexagons at distance 3)
    # We take 5 of them
    r2 = 3.0
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    # Remove 270 (index 4)
    angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for angle in angles:
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 4.0, 0.0
