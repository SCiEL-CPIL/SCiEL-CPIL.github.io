"""Rank-189 decomposition for 6x6 matrix multiplication using 3x3 block Strassen."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 6
    start_time = time.time()
    
    # 2x2 Strassen coefficients (fixed)
    strassen_A = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_B = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    # Rank 189: 3x3 blocks of 2x2, each 2x2 product uses rank-7 Strassen
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for t in range(7):
                    for (ai, ak), coeff in strassen_A[t].items():
                        global_i = 2 * bi + ai
                        global_k = 2 * bj + ak
                        U[global_i * m + global_k][col] += coeff
                    for (bk_local, bj_local), coeff in strassen_B[t].items():
                        global_bk = 2 * bj + bk_local
                        global_bj = 2 * bk + bj_local
                        V[global_bk * p + global_bj][col] += coeff
                    for (ci, cj), coeff in strassen_W[t]:
                        global_i = 2 * bi + ci
                        global_j = 2 * bk + cj
                        W[global_j * n + global_i][col] += coeff
                    col += 1
    
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Safety: return what we have (shouldn't happen for fixed construction)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
