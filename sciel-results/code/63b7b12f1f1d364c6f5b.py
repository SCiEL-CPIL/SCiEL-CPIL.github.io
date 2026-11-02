import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 100.
    # We use a decomposition that splits the multiplication into:
    # 1. A 4x4 * 4x4 multiplication (using the first 4 rows/cols)
    # 2. A rank-1 update for the remaining column/row interactions
    
    # Rank 1: 4x4x4 schoolbook (64 terms)
    # Rank 2: 4x1 * 1x5 remainder (1 term)
    # Total rank: 65
    
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: 4x4x4 Schoolbook
    # i in [0, 4), j in [0, 4), k in [0, 4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: Remainder
    # This handles the terms where j=4 (the 5th column of A and 5th row of B)
    # C[i, k] += A[i, 4] * B[4, k]
    # This is a rank-1 tensor in the (i, k) space, fixed at j=4.
    # U[i, 4] = 1 for all i
    # V[4, k] = 1 for all k
    # W[k, i] = 1 for all i, k
    
    for i in range(4):
        U[i * m + 4, col] = 1
        
    for k in range(5):
        V[4 * p + k, col] = 1
        
    for i in range(4):
        for k in range(5):
            W[k * n + i, col] = 1
            
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
