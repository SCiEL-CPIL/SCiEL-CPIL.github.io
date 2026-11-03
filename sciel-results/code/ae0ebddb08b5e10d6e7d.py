import math

def construct():
    N = 11
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 neighbors
    for k in range(6):
        angle = math.pi / 3 * k
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # 4 midpoints of the next shell
    # Angles: 30, 90, 210, 270 degrees
    angles = [math.pi/6, math.pi/2, math.pi + math.pi/6, math.pi + math.pi/2]
    for angle in angles:
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 3.0, 90.0
