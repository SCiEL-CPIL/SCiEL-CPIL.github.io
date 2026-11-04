import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Local indices i', j', k' in {0, 1}
    # Global i = i', j = j', k = k_start + k'
    
    # M1: (a+d)(e+h)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: dict (i', j') -> val
        # V: dict (j', k') -> val
        # W: dict (k', i') -> val
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}), # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}), # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})  # M7
    ]
    
    r = 0
    for b in range(3):
        k_start = 2 * b
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            for (ip, jp), val in u_coeffs.items():
                U[ip * m + jp][r] += val
            for (jp, kp), val in v_coeffs.items():
                V[jp * p + (k_start + kp)][r] += val
            for (kp, ip), val in w_coeffs.items():
                W[(k_start + kp) * n + ip][r] += val
            r += 1
            
    # Rank-1 terms for 2x1x2 (j=2)
    for b in range(3):
        k_start = 2 * b
        # U: (i, 2) = 1 for i in {0, 1}
        for i in range(2):
            U[i * m + 2][r] = 1
        # V: (2, k) = 1 for k in {k_start, k_start+1}
        for k in range(2):
            V[2 * p + (k_start + k)][r] = 1
        # W: (k, i) = 1 for k in {k_start, k_start+1}, i in {0, 1}
        for k in range(2):
            for i in range(2):
                W[(k_start + k) * n + i][r] = 1
        r += 1
        
    return {"U": U, "V": V, "W": W}
