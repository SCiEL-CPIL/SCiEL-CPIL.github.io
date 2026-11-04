import math

N = 11

def construct():
    # 1 central hexagon
    inner = [[0.0, 0.0, 0.0]]
    
    # 6 hexagons in the first ring (distance sqrt(3))
    dist1 = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        x = dist1 * math.cos(angle)
        y = dist1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # 4 hexagons from the second ring (distance 3)
    # We pick 4 out of the 6 available at distance 3.
    # To keep it simple, we pick 4.
    dist2 = 3.0
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    # Let's pick 30, 90, 150, 210
    angles2 = [30, 90, 150, 210]
    for deg in angles2:
        angle = deg * math.pi / 180
        x = dist2 * math.cos(angle)
        y = dist2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 4.0, 0.0
