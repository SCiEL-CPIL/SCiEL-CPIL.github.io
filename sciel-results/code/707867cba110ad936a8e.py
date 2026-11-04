"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 140

import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # 1. Define Strassen 2x2x2 decomposition (Rank 7)
    # U, V, W are 4x7 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 1, 1, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # 2. Construct 4x4x4 decomposition (Rank 49) via Kronecker product
    # U4, V4, W4 are 16x49 matrices
    # Index mapping:
    # i = ih*2 + il, j = jh*2 + jl, k = kh*2 + kl
    # r = rh*7 + rl
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for ih in range(2):
        for il in range(2):
            for jh in range(2):
                for jl in range(2):
                    for kh in range(2):
                        for kl in range(2):
                            i = ih*2 + il
                            j = jh*2 + jl
                            k = kh*2 + kl
                            
                            u_idx = i*4 + j
                            v_idx = j*4 + k
                            w_idx = k*4 + i
                            
                            for rh in range(7):
                                for rl in range(7):
                                    r = rh*7 + rl
                                    
                                    U4[u_idx, r] += U2[ih*2+jh, rh] * U2[il*2+jl, rl]
                                    V4[v_idx, r] += V2[jh*2+kh, rh] * V2[jl*2+kl, rl]
                                    W4[w_idx, r] += W2[kh*2+ih, rh] * W2[kl*2+il, rl]
                                    
    # 3. Construct 4x4x8 decomposition (Rank 98)
    # Split B into two 4x4 blocks B1, B2.
    # C1 = A B1, C2 = A B2.
    # U is 16x98. First 49 cols for C1, next 49 for C2.
    # V is 32x98. 
    #   For C1: V rows correspond to B1 (rows 0-3, cols 0-3 of original B).
    #   For C2: V rows correspond to B2 (rows 0-3, cols 4-7 of original B).
    # W is 32x98.
    #   For C1: W rows correspond to C1 (rows 0-3 of original C).
    #   For C2: W rows correspond to C2 (rows 4-7 of original C).
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First block (C1)
    U[:, 0:49] = U4
    
    # Map V4 (16x49) to V (32x98)
    # V4 index: j*4 + k (j in 0..3, k in 0..3)
    # V index: j*8 + k (j in 0..3, k in 0..3)
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
            
    # Map W4 (16x49) to W (32x98)
    # W4 index: k*4 + i (k in 0..3, i in 0..3)
    # W index: k*4 + i (k in 0..3, i in 0..3)
    W[0:16, 0:49] = W4
    
    # Second block (C2)
    U[:, 49:98] = U4
    
    # Map V4 to V for C2
    # V4 index: j*4 + k (j in 0..3, k in 0..3)
    # V index: j*8 + (k+4) (j in 0..3, k in 0..3)
    for j in range(4):
        for k in range(4):
            V[j*8 + (k+4), 49:98] = V4[j*4 + k, :]
            
    # Map W4 to W for C2
    # W4 index: k*4 + i (k in 0..3, i in 0..3)
    # W index: (k+4)*4 + i (k in 0..3, i in 0..3)
    W[16:32, 49:98] = W4
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
