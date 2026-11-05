import math

def construct():
    # 11 unit hexagons
    # 7 in a flower pattern
    # 4 in the gaps (spikes)
    
    sqrt3 = math.sqrt(3)
    
    # Centers
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0.0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5),
        # Spikes
        (1.5*sqrt3, 1.5), # Gap 30 deg
        (0.0, 3.0),       # Gap 90 deg
        (-1.5*sqrt3, 1.5),# Gap 150 deg
        (0.0, -3.0)       # Gap 270 deg
    ]
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length 4.0
    # Centered at 0,0
    # Rotation 90 degrees (vertex up) to align with spikes
    # The spikes are at 90 and 270, so vertex up is good.
    return inner, (0.0, 0.0), 4.0, 90.0
