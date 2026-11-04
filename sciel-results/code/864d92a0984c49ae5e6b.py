"""
Strategy: Block Composition (Corrected).
Decompose 4x4x9 into two 4x4x4 (Strassen, rank 49) and one 4x4x1 (Schoolbook, rank 16).
Total Rank = 114.
This is better than schoolbook 144.
"""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # 1. Strassen 2x2x2 (Rank 7)
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ])
    V2 = np.array([
        [1, -1, 0, 1, -1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ])
    
    # 2. Construct 4x4x4 Strassen (Rank 49) via Kronecker
    r4 = 49
    U4 = np.zeros((16, r4))
    V4 = np.zeros((16, r4))
    W4 = np.zeros((16, r4))
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            for c1 in range(7):
                                for c2 in range(7):
                                    idx_c = c1*7 + c2
                                    
                                    u1 = U2[i1*2+j1][c1]
                                    u2 = U2[i2*2+j2][c2]
                                    v1 = V2[j1*2+k1][c1]
                                    v2 = V2[j2*2+k2][c2]
                                    w1 = W2[k1*2+i1][c1]
                                    w2 = W2[k2*2+i2][c2]
                                    
                                    idx_u = (i1*2+i2)*4 + (j1*2+j2)
                                    idx_v = (j1*2+j2)*4 + (k1*2+k2)
                                    idx_w = (k1*2+k2)*4 + (i1*2+i2)
                                    
                                    U4[idx_u][idx_c] += u1 * u2
                                    V4[idx_v][idx_c] += v1 * v2
                                    W4[idx_w][idx_c] += w1 * w2
    
    # 3. Construct 4x4x1 Schoolbook (Rank 16)
    r1 = 16
    U1 = np.zeros((16, r1))
    V1 = np.zeros((4, r1))  # 4x1
    W1 = np.zeros((4, r1))  # 1x4 -> W index k*n+i. k=0.
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i*4+j][col] = 1
            V1[j][col] = 1 # V index j*p+k. p=1, k=0.
            W1[i][col] = 1 # W index k*n+i. k=0.
            col += 1

    # 4. Assemble 4x4x9
    R = 114
    U = np.zeros((16, R))
    V = np.zeros((36, R)) # 4x9
    W = np.zeros((36, R)) # 9x4
    
    # Block 1: k=0,1,2,3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*9+k, 0:49] = V4[j*4+k]
    for i in range(4):
        for k in range(4):
            W[k*4+i, 0:49] = W4[k*4+i]
            
    # Block 2: k=4,5,6,7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*9+k, 49:98] = V4[j*4+(k-4)]
    for i in range(4):
        for k in range(4, 8):
            W[k*4+i, 49:98] = W4[(k-4)*4+i]
            
    # Block 3: k=8
    U[:, 98:114] = U1
    for j in range(4):
        V[j*9+8, 98:114] = V1[j]
    for i in range(4):
        W[8*4+i, 98:114] = W1[i]
        
    # Convert to lists for output
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
