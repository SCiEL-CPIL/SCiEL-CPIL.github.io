import numpy as np

def construct():
    # Strategy: Concentric ring packing with 11 hexagons
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons at distance sqrt(3) (touching neighbors)
    # Ring 2: 4 hexagons at distance 3 (filling gaps in ring 1)
    # All inner hexagons rotated 30 degrees (flat-top orientation)
    # Outer hexagon also rotated 30 degrees
    
    centers = []
    
    # Ring 0: center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3), angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3, angles 30, 90, 150, 270
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Calculate required outer side length
    max_x = 0.0
    max_y = 0.0
    
    for x, y in centers:
        max_x = max(max_x, x + np.sqrt(3)/2)
        max_y = max(max_y, y + 1.0)
        max_x = max(max_x, -x + np.sqrt(3)/2)
        max_y = max(max_y, -y + 1.0)
        
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    
    side = max(s_x, s_y)
    
    return inner, (0.0, 0.0), side, 30.0
