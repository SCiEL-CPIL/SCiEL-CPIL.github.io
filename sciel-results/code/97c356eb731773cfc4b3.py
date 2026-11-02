"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
_start_time = time.time()
_DEADLINE = 120.0

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    rank = 7
    
    # Strassen decomposition for 2x2x2
    # u_r[i*m+j], v_r[j*p2+k2], w_r[k2*n+i] for 2x2x2
    # p2=2, n=2
    
    # u_r: coeff of A[i,j] in M_r
    # v_r: coeff of B[j,k] in M_r
    # w_r: coeff of M_r in C[k,i]
    
    u = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ]
    
    v = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [-1, 0, 1, 0], # M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    w = [
        [1, 0, 0, 1], # M1: C00+C11 (idx 0, 3)
        [0, 0, 1, -1], # M2: C10-C11 (idx 2, 3)
        [0, 0, 1, 1], # M3: C01+C11 (idx 2, 3)
        [1, 0, 1, 0], # M4: C00+C10 (idx 0, 2)
        [-1, 0, 1, 0], # M5: -C00+C01 (idx 0, 2)
        [0, 0, 0, 1], # M6: C11 (idx 3)
        [1, 0, 0, 0]  # M7: C00 (idx 0)
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # Fill U
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                U[idx][r] = u[r][idx]
        
        # Fill V
        for j in range(m):
            for k in range(p):
                k2 = k % 2
                idx_v = j * p + k
                idx_v2 = j * 2 + k2
                V[idx_v][r] = v[r][idx_v2]
        
        # Fill W
        for k in range(p):
            k2 = k % 2
            for i in range(n):
                idx_w = k * n + i
                idx_w2 = k2 * n + i
                W[idx_w][r] = w[r][idx_w2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
