import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2x2
    # u: list of (i, j_prime, val)
    # v: list of (j_prime, k_prime, val)
    # c: list of (i, k_prime, val)
    
    strassen_terms = [
        # M1
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # M2
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        # M3
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        # M4
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        # M5
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        # M6
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # M7
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)]
        )
    ]
    
    col = 0
    for jb in range(2):
        for kb in range(4):
            for u_list, v_list, c_list in strassen_terms:
                for i, jp, val in u_list:
                    row_u = i * m + (2 * jb + jp)
                    U[row_u][col] = val
                for jp, kp, val in v_list:
                    row_v = (2 * jb + jp) * p + (2 * kb + kp)
                    V[row_v][col] = val
                for i, kp, val in c_list:
                    row_w = (2 * kb + kp) * n + i
                    W[row_w][col] = val
                col += 1
                
    return {"U": U, "V": V, "W": W}
