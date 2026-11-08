import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Lattice basis vectors for aligned hexagons
    # v0 = (1.5, sqrt(3)/2), v1 = (0, sqrt(3))
    # c = u * v0 + v * v1
    # x = 1.5 * u
    # y = sqrt(3)/2 * (u + 2*v)
    
    sqrt3 = math.sqrt(3)
    
    # Coordinates (u, v) of 13 points that fit in S'=3 hexagon
    # We pick 12 of them.
    points_uv = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
        (-1, -1),
        (2, -1),
        (-2, 1),
        (1, -2)
        # (-1, 2) is the 13th, we omit it
    ]
    
    inner = []
    for u, v in points_uv:
        x = 1.5 * u
        y = (sqrt3 / 2) * (u + 2 * v)
        inner.append([x, y, 0.0])
        
    # Outer hexagon side length
    # S' = 3, so S = S' + 1 = 4
    side = 4.0
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
