import math

N = 12

def construct():
    # Rotation 30 degrees
    angle = 30.0
    
    # Spacing
    dx = math.sqrt(3)
    dy = 2.0
    
    # 6 hexagons per row
    # Centers x: -2.5*dx to 2.5*dx
    x_centers = [(k - 2.5) * dx for k in range(6)]
    
    inner = []
    # Row 1 (Top)
    for x in x_centers:
        inner.append([x, 1.0, angle])
        
    # Row 2 (Bottom)
    for x in x_centers:
        inner.append([x, -1.0, angle])
        
    S = 19.0 * math.sqrt(3) / 6.0
    
    return inner, (0.0, 0.0), S, 0.0
