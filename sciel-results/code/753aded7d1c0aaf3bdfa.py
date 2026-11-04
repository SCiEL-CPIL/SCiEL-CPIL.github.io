import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split A (4x5) into A1 (4x4) and A2 (4x1)
    # Split B (5x5) into B1 (4x5) and B2 (1x5)
    # C = A1 B1 + A2 B2
    # A1 B1: 4x4 * 4x5
    # Split B1 (4x5) into B11 (4x4) and B12 (4x1)
    # A1 B11: 4x4 * 4x4 (Rank 64)
    # A1 B12: 4x4 * 4x1 (Rank 16)
    # A2 B2: 4x1 * 1x5 (Rank 1)
    
    # Total Rank = 64 + 16 + 1 = 81
    
    rank = 81
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A1 B11 (4x4 * 4x4)
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Part 2: A1 B12 (4x4 * 4x1)
    # i in 0..3, j in 0..3, k in 4..4
    for i in range(4):
        for j in range(4):
            for k in range(4, 5):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Part 3: A2 B2 (4x1 * 1x5)
    # i in 0..3, j in 4..4, k in 0..4
    for i in range(4):
        for j in range(4, 5):
            for k in range(5):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
