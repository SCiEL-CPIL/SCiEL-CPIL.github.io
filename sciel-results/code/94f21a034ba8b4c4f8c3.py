import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Rank 28 decomposition using 4x Strassen 2x2x2
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Sub-product definitions
    # Each sub-product is a 2x2 matrix multiplication
    # P indices in U, Q indices in V, R indices in W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Sub 0: A_L (cols 0,1) * B_TL (rows 0,1; cols 0,1) -> C_Left (cols 0,1)
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0, 1, 4, 5
    # B_TL: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0, 1, 4, 5
    # C_Left: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0, 2, 1, 3
    sub0 = {
        'u': [0, 1, 4, 5],
        'v': [0, 1, 4, 5],
        'w': [0, 2, 1, 3],
        'base': 0
    }
    
    # Sub 1: A_R (cols 2,3) * B_BL (rows 2,3; cols 0,1) -> C_Left (cols 0,1)
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2, 3, 6, 7
    # B_BL: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 8, 9, 12, 13
    # C_Left: Same W indices
    sub1 = {
        'u': [2, 3, 6, 7],
        'v': [8, 9, 12, 13],
        'w': [0, 2, 1, 3],
        'base': 7
    }
    
    # Sub 2: A_L (cols 0,1) * B_TR (rows 0,1; cols 2,3) -> C_Right (cols 2,3)
    # A_L: Same U indices
    # B_TR: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2, 3, 6, 7
    # C_Right: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices 4, 6, 5, 7
    sub2 = {
        'u': [0, 1, 4, 5],
        'v': [2, 3, 6, 7],
        'w': [4, 6, 5, 7],
        'base': 14
    }
    
    # Sub 3: A_R (cols 2,3) * B_BR (rows 2,3; cols 2,3) -> C_Right (cols 2,3)
    # A_R: Same U indices
    # B_BR: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 10, 11, 14, 15
    # C_Right: Same W indices
    sub3 = {
        'u': [2, 3, 6, 7],
        'v': [10, 11, 14, 15],
        'w': [4, 6, 5, 7],
        'base': 21
    }
    
    subs = [sub0, sub1, sub2, sub3]
    
    # Strassen coefficients for M_k (k=1..7)
    # M1 = (p00 + p11)(q00 + q11)
    # M2 = (p10 + p11)q00
    # M3 = p00(q01 - q11)
    # M4 = p11(q10 - q00)
    # M5 = (p00 + p01)q11
    # M6 = (p10 - p00)(q00 + q01)
    # M7 = (p01 - p11)(q10 + q11)
    
    # Result
    # r00 = M1 + M4 - M5 + M7
    # r01 = M3 + M5
    # r10 = M2 + M4
    # r11 = M1 - M2 + M3 + M6
    
    strassen_u = [
        [1, 0, 1, 0, 1, -1, 0], # p00
        [0, 0, 0, 0, 1, 0, 1],  # p01
        [0, 1, 0, 0, 0, 1, 0],  # p10
        [1, 1, 0, 1, 0, 0, -1]  # p11
    ]
    
    strassen_v = [
        [1, 1, 0, -1, 0, 1, 0], # q00
        [0, 0, 1, 0, 0, 1, 0],  # q01
        [0, 0, -1, 1, 0, 0, 1], # q10
        [1, 0, 0, 0, 1, 0, 1]   # q11
    ]
    
    strassen_w = [
        [1, 0, 0, 1, -1, 0, 1], # r00
        [0, 0, 1, 0, 1, 0, 0],  # r01
        [0, 1, 0, 1, 0, 0, 0],  # r10
        [1, -1, 1, 0, 0, 1, 0]  # r11
    ]
    
    for sub in subs:
        base = sub['base']
        u_idx = sub['u']
        v_idx = sub['v']
        w_idx = sub['w']
        
        for k in range(7):
            col = base + k
            # Update U
            for pi in range(4):
                coeff = strassen_u[pi][k]
                if coeff != 0:
                    U[u_idx[pi]][col] += coeff
                    
            # Update V
            for qi in range(4):
                coeff = strassen_v[qi][k]
                if coeff != 0:
                    V[v_idx[qi]][col] += coeff
                    
            # Update W
            for ri in range(4):
                coeff = strassen_w[ri][k]
                if coeff != 0:
                    W[w_idx[ri]][col] += coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
