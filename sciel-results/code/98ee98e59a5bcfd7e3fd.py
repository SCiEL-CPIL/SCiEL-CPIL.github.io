import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Axial coordinates for 11 hexagons
    # 9 points that fit in hexagon of side 2 (approx)
    # Plus 2 points extending to side 2*sqrt(3)
    axial = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (-1, 1),
        (0, -1),
        (1, -1),
        (-1, 2),
        (1, -2),
        (-2, 0),
        (2, 0)
    ]
    
    inner = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # The outer hexagon side length
    # S_inner = 2 * sqrt(3)
    # S_outer = S_inner + 1
    side = 2 * math.sqrt(3) + 1.0
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
