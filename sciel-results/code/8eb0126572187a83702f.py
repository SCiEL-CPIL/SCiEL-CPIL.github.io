import math

def construct():
    # N = 11
    # We pack 11 unit hexagons in a hexagon of side 4.
    # The centers are on a hex grid with spacing sqrt(3).
    # Coordinates (q, r) in axial system:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # The valid (q, r) pairs for a container of side L=3 (which corresponds to S=4)
    # are:
    # r=1: q in {-2, -1, 0, 1}
    # r=0: q in {-1, 0, 1}
    # r=-1: q in {-2, -1, 0, 1}
    
    inner = []
    sqrt3 = math.sqrt(3)
    
    for r in [1, 0, -1]:
        if r == 0:
            qs = [-1, 0, 1]
        else:
            qs = [-2, -1, 0, 1]
            
        for q in qs:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            # All hexagons are aligned with angle 0
            inner.append([x, y, 0.0])
            
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
