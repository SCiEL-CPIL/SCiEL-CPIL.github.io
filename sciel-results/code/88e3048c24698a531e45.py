import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U_idx: (i, j) -> i*m + j
    # V_idx: (j, k) -> j*p + k
    # W_idx: (k, i) -> k*n + i
    
    # Terms defined by (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (i, j, val)
    # V_coeffs: list of (j, k_local, val)
    # W_coeffs: list of (k_local, i, val)
    
    terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(1,0,1), (1,1,-1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (1,1,-1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1), (0,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (1,0,1)])
    ]
    
    for l in range(5):
        for s in range(7):
            col = l * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for i, j, val in u_coeffs:
                U[i * m + j][col] += val
                
            for j, k_local, val in v_coeffs:
                k_global = 2 * l + k_local
                V[j * p + k_global][col] += val
                
            for k_local, i, val in w_coeffs:
                k_global = 2 * l + k_local
                W[k_global * n + i][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
