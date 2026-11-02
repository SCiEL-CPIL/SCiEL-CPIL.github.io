import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2
    U_222 = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0]
    ], dtype=float)
    
    V_222 = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1]
    ], dtype=float)
    
    W_222 = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=float)
    
    # Rank 4 decomposition for 2x2x1 (Schoolbook)
    # U is 4x4, V is 4x4, W is 4x4
    # T[i,j,k] = 1. k is always 0.
    # U[i,j] = 1, V[j,0] = 1, W[0,i] = 1
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((4, 4))
    W_221 = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            U_221[idx, idx] = 1
            V_221[idx, idx] = 1
            W_221[i, idx] = 1 # W[k*n+i] -> W[0*2+i] = W[i]
            
    # Construct 2x2x5 from two 2x2x2 and one 2x2x1
    # Rank = 7 + 7 + 4 = 18
    U_225 = np.hstack([U_222, U_222, U_221])
    V_225 = np.hstack([V_222, V_222, V_221])
    W_225 = np.hstack([W_222, W_222, W_221])
    
    # Construct 2x4x5 from two 2x2x5 blocks (splitting m=4 into 2+2)
    # A is 2x4, split into A_L (2x2) and A_R (2x2)
    # B is 4x5, split into B_T (2x5) and B_B (2x5)
    # C = A_L B_T + A_R B_B
    
    # U for 2x4x5 is size (2*4) x Rank = 8 x 36
    # Block 1 (A_L B_T): U indices i in 0..1, j in 0..1.
    # Block 2 (A_R B_B): U indices i in 0..1, j in 2..3.
    
    rank_225 = U_225.shape[1]
    total_rank = 2 * rank_225
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1: i in 0..1, j in 0..1, k in 0..4
    # U indices: i*m + j. j is 0,1.
    # V indices: j*p + k. j is 0,1.
    # W indices: k*n + i.
    
    for c in range(rank_225):
        # U part 1
        for i in range(2):
            for j in range(2):
                U[i * m + j, c] = U_225[i * 2 + j, c]
                
        # V part 1
        for j in range(2):
            for k in range(5):
                V[j * p + k, c] = V_225[j * 5 + k, c]
                
        # W part 1
        for k in range(5):
            for i in range(2):
                W[k * n + i, c] = W_225[k * 2 + i, c]
                
    # Block 2: i in 0..1, j in 2..3, k in 0..4
    # Map j_local = j - 2 (0,1)
    # U indices: i*m + j.
    # V indices: j*p + k.
    # W indices: k*n + i.
    
    for c in range(rank_225):
        col_offset = rank_225 + c
        
        # U part 2
        for i in range(2):
            for j_local in range(2):
                j = j_local + 2
                U[i * m + j, col_offset] = U_225[i * 2 + j_local, c]
                
        # V part 2
        for j_local in range(2):
            j = j_local + 2
            for k in range(5):
                V[j * p + k, col_offset] = V_225[j_local * 5 + k, c]
                
        # W part 2
        for k in range(5):
            for i in range(2):
                W[k * n + i, col_offset] = W_225[k * 2 + i, c]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
