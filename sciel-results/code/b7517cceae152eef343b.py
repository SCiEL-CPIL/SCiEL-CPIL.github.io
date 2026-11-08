import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 11 unit hexagons
    # 1 at center, 6 in first ring, 4 in second ring
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons)
    # Distance sqrt(3)
    dist1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = dist1 * math.cos(angle)
        y = dist1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Second ring (4 hexagons)
    # Distance 2*sqrt(3)
    dist2 = 2 * math.sqrt(3)
    # Place at 0, 60, 120, 180 degrees
    for i in range(4):
        angle = math.radians(60 * i)
        x = dist2 * math.cos(angle)
        y = dist2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length 5.0
    center = (0.0, 0.0)
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
