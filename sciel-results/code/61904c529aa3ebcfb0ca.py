import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS

    # Build rank-49 Strassen decomposition (7 outer block multiplications, each 7 inner Strassen products)
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Outer Strassen 2x2 block formula
    outer_defs = [
        {'a': [(0,0,1),(1,1,1)], 'b': [(0,0,1),(1,1,1)], 'c': [(0,0,1),(1,1,1)]},
        {'a': [(1,0,1),(1,1,1)], 'b': [(0,0,1)],          'c': [(1,0,1),(1,1,-1)]},
        {'a': [(0,0,1)],           'b': [(0,1,1),(1,1,-1)],'c': [(0,1,1),(1,1,1)]},
        {'a': [(1,1,1)],           'b': [(1,0,1),(0,0,-1)],'c': [(0,0,1),(1,0,1)]},
        {'a': [(0,0,1),(0,1,1)],   'b': [(1,1,1)],          'c': [(0,1,1),(0,0,-1)]},
        {'a': [(1,0,1),(0,0,-1)],  'b': [(0,0,1),(0,1,1)],  'c': [(1,1,1)]},
        {'a': [(0,1,1),(1,1,-1)],  'b': [(1,0,1),(1,1,1)],  'c': [(0,0,1)]},
    ]

    # Inner Strassen 2x2 multiplication
    inner_defs = [
        {'x': [(0,0,1),(1,1,1)],  'y': [(0,0,1),(1,1,1)],  'z': [(0,0,1),(1,1,1)]},
        {'x': [(1,0,1),(1,1,1)],  'y': [(0,0,1)],           'z': [(1,0,1),(1,1,-1)]},
        {'x': [(0,0,1)],           'y': [(0,1,1),(1,1,-1)], 'z': [(0,1,1),(1,1,1)]},
        {'x': [(1,1,1)],           'y': [(1,0,1),(0,0,-1)], 'z': [(0,0,1),(1,0,1)]},
        {'x': [(0,0,1),(0,1,1)],   'y': [(1,1,1)],          'z': [(0,1,1),(0,0,-1)]},
        {'x': [(1,0,1),(0,0,-1)],  'y': [(0,0,1),(0,1,1)],  'z': [(1,1,1)]},
        {'x': [(0,1,1),(1,1,-1)],  'y': [(1,0,1),(1,1,1)],  'z': [(0,0,1)]},
    ]

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t
            for (Pb, Qb, so) in outer['a']:
                for (xi, xj, si) in inner['x']:
                    U[2*Pb+xi][2*Qb+xj] = 0  # placeholder, will compute below
                    U[(2*Pb+xi)*m + (2*Qb+xj)][r] += so * si
            for (Pb, Qb, so) in outer['b']:
                for (yj, yk, si) in inner['y']:
                    V[(2*Pb+yj)*p + (2*Qb+yk)][r] += so * si
            for (Pb, Qb, so) in outer['c']:
                for (zi, zk, si) in inner['z']:
                    W[(2*Qb+zk)*n + (2*Pb+zi)][r] += so * si

    return {"U": U, "V": V, "W": W}
