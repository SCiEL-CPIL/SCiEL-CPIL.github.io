import math

def construct():
    N = 11
    sqrt3 = math.sqrt(3)
    S = 1.0 + 2.0 * sqrt3
    
    # Lattice points (i, j)
    points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    inner = []
    for i, j in points:
        x = sqrt3 * (i + j / 2.0)
        y = 1.5 * j
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
