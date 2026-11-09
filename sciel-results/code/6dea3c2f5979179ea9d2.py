import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    inner = []
    
    # 1 center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 neighbors at distance sqrt(3)
    d1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # 5 more at distance 3, angles 30, 90, 150, 210, 270
    d2 = 3.0
    angles = [30, 90, 150, 210, 270]
    for deg in angles:
        angle = math.radians(deg)
        x = d2 * math.cos(angle)
        y = d2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle_deg = 30.0
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END
