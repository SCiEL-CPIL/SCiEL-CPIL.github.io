import math

N = 11

def construct():
    # 7-hexagon cluster centers
    # Center at (0,0)
    centers = [(0.0, 0.0)]
    # 6 neighbors at distance 2
    for i in range(6):
        angle = math.radians(60 * i)
        x = 2.0 * math.cos(angle)
        y = 2.0 * math.sin(angle)
        centers.append((x, y))
    
    # 4 extensions on sides
    # Sides are at angles 30, 90, 150, 210, 270, 330
    # We pick 4 of them, e.g., 30, 150, 210, 330
    ext_angles = [30, 150, 210, 330]
    for deg in ext_angles:
        angle = math.radians(deg)
        # Distance from center to side of 7-cluster is 3 * sqrt(3) / 2
        # Distance from side to center of new hex is sqrt(3) / 2
        # Total distance = 3 * sqrt(3) / 2 + sqrt(3) / 2 = 2 * sqrt(3)
        dist = 2.0 * math.sqrt(3)
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        centers.append((x, y))
        
    # Total 11 centers
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Calculate required S
    # Max extent is 2*sqrt(3) + 1
    S = 2.0 * math.sqrt(3) + 1.0 + 0.01 # small margin
    
    return inner, (0.0, 0.0), S, 0.0
