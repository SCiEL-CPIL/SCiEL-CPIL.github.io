import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    # 7 for (2x2) * (2x2) [k=0,1]
    # 7 for (2x2) * (2x2) [k=2,3]
    # 1 for (2x1) * (1x4) [j=2]
    R = 15
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # U_coeffs: list of (i, j, coeff)
    # V_coeffs: list of (j, k, coeff)
    # W_coeffs: list of (i, k, coeff)
    
    strassen_terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(1,0,1), (0,0,-1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,-1)])
    ]
    
    col = 0
    # Block 1: k=0,1
    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
        for i, j, c in u_coeffs:
            U[i * m + j, col] += c
        for j, k, c in v_coeffs:
            V[j * p + k, col] += c
        for i, k, c in w_coeffs:
            W[k * n + i, col] += c
        col += 1
        
    # Block 2: k=2,3
    # Shift k in V and W by +2
    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
        for i, j, c in u_coeffs:
            U[i * m + j, col] += c
        for j, k, c in v_coeffs:
            V[j * p + (k + 2), col] += c
        for i, k, c in w_coeffs:
            W[(k + 2) * n + i, col] += c
        col += 1
            
    # Block 3: j=2, all k
    # U[i, 2] = 1
    # V[2, k] = 1
    # W[k, i] = 1
    for i in range(n):
        U[i * m + 2, col] = 1
    for k in range(p):
        V[2 * p + k, col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, col] = 1
    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
