import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # u: 4 entries (A00, A01, A10, A11)
    # v: 4 entries (B00, B01, B10, B11)
    # w: 4 entries (C00, C01, C10, C11)
    
    s_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    s_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    s_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 1, 0], # M4
        [0, 1, 0, 1], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    # Blocks: (i_range, j_range, k_range)
    # i: 0,1
    # j: 0,1,2,3
    # k: 0,1,2,3
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3])
    ]
    
    col = 0
    for i_range, j_range, k_range in blocks:
        i0 = i_range[0]
        j0 = j_range[0]
        k0 = k_range[0]
        
        for r in range(7):
            # Fill U
            for li in range(2):
                for lj in range(2):
                    gi = i0 + li
                    gj = j0 + lj
                    U[gi * m + gj][col] = s_u[r][li * 2 + lj]
                    
            # Fill V
            for lj in range(2):
                for lk in range(2):
                    gj = j0 + lj
                    gk = k0 + lk
                    V[gj * p + gk][col] = s_v[r][lj * 2 + lk]
                    
            # Fill W
            for lk in range(2):
                for li in range(2):
                    gk = k0 + lk
                    gi = i0 + li
                    W[gk * n + gi][col] = s_w[r][lk * 2 + li]
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
