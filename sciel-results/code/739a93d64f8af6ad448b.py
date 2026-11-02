"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Define the 7 rank-1 tensors for 2x2x2 Strassen decomposition
    # U, V, W are vectors of length 4 (flattened 2x2 matrices)
    # Indexing: 00, 01, 10, 11
    
    U2 = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    V2 = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    W2 = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: -C00+C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    rank = 98
    U = np.zeros((n * m, rank), dtype=float) # 16 x 98
    V = np.zeros((m * p, rank), dtype=float) # 32 x 98
    W = np.zeros((p * n, rank), dtype=float) # 32 x 98
    
    # Helper to flatten 2x2 to 4-vector index
    def idx2(i, j):
        return i * 2 + j
        
    # Build the 49 terms for one 4x4x4 multiplication
    for s in range(7):
        for t in range(7):
            # Generate U4, V4, W4 for this (s,t) pair
            # U4 is 4x4, flattened to 16
            U4 = np.zeros(16)
            V4 = np.zeros(16)
            W4 = np.zeros(16)
            
            for I in range(4):
                for J in range(4):
                    r = I // 2
                    c = J // 2
                    i_loc = I % 2
                    j_loc = J % 2
                    coeff = U2[s][idx2(r, c)] * U2[t][idx2(i_loc, j_loc)]
                    U4[I * 4 + J] = coeff
                    
            for J in range(4):
                for K in range(4):
                    r = J // 2
                    c = K // 2
                    j_loc = J % 2
                    k_loc = K % 2
                    coeff = V2[s][idx2(r, c)] * V2[t][idx2(j_loc, k_loc)]
                    V4[J * 4 + K] = coeff
                    
            for I in range(4):
                for K in range(4):
                    r = I // 2
                    c = K // 2
                    i_loc = I % 2
                    k_loc = K % 2
                    coeff = W2[s][idx2(r, c)] * W2[t][idx2(i_loc, k_loc)]
                    W4[I * 4 + K] = coeff # Note: W4 index is I, K. W matrix index is K, I.
            
            # Left part (cols 0-3)
            col_idx = s * 7 + t
            U[:, col_idx] = U4
            for j in range(4):
                for k in range(4):
                    V[j * 8 + k, col_idx] = V4[j * 4 + k]
            for k in range(4):
                for i in range(4):
                    W[k * 4 + i, col_idx] = W4[i * 4 + k]
                    
            # Right part (cols 4-7)
            col_idx_right = 49 + s * 7 + t
            U[:, col_idx_right] = U4
            for j in range(4):
                for k in range(4):
                    V[j * 8 + (k + 4), col_idx_right] = V4[j * 4 + k]
            for k in range(4):
                for i in range(4):
                    W[(k + 4) * 4 + i, col_idx_right] = W4[i * 4 + k]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
