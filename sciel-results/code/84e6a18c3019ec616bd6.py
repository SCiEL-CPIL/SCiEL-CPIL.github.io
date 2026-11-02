import math

N = 11

def construct():
    # Arrangement: 3-5-3 rows
    # dx = 2 * sqrt(3)
    # dy = 3
    dx = 2.0 * math.sqrt(3.0)
    dy = 3.0
    
    inner = []
    
    # Row 1 (middle, y=0): 5 hexagons
    for k in range(-2, 3):
        x = k * dx
        y = 0.0
        angle = 30.0
        inner.append([x, y, angle])
        
    # Row 0 (top, y=dy): 3 hexagons
    for k in range(-1, 2):
        x = k * dx
        y = dy
        angle = 30.0
        inner.append([x, y, angle])
        
    # Row 2 (bottom, y=-dy): 3 hexagons
    for k in range(-1, 2):
        x = k * dx
        y = -dy
        angle = 30.0
        inner.append([x, y, angle])
        
    # Outer hexagon
    # Side length S = 4.5 * sqrt(3)
    side = 4.5 * math.sqrt(3.0)
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
