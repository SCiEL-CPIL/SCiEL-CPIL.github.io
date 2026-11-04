import math

N = 11

def construct():
    # Generate positions for a hexagonal cluster of side 2 (19 hexagons)
    # and pick 11 of them.
    # The 19-cluster fits in a hexagon of side 4.5 (radius ~4.464).
    
    centers = []
    # Layer 0
    centers.append((0.0, 0.0))
    
    # Layer 1 (6 hexagons)
    for i in range(6):
        angle = i * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # Layer 2 (12 hexagons)
    # We only need 4 more to reach 11.
    # Let's pick 4 from Layer 2.
    # To be safe, we pick them such that they are well within side 4.5.
    # Actually, any 4 from Layer 2 will be at distance 2*sqrt(3) from center.
    # Max distance = 2*sqrt(3) + 1 = 4.464 < 4.5.
    
    for i in range(4):
        # Place them at angles 0, 90, 180, 270?
        # Layer 2 angles are 30, 90, 150... (offset by 30 from Layer 1)
        # Let's just pick 4 consecutive ones.
        angle = (i * 60 + 30) * math.pi / 180
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # We have 1 + 6 + 4 = 11 centers.
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 4.5, 0.0
