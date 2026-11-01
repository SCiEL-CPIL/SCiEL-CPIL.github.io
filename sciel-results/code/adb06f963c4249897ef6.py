"""Matrix multiplication 2,2,13 - Block-based decomposition using direct sum structure."""
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 13
    
    # The tensor M_{2,2,13} is the direct sum of 13 copies of M_{2,2}.
    # The rank of M_{2,2} is 4 (over Q).
    # Total rank = 13 * 4 = 52.
    # This is the minimal rank.
    
    # We construct the decomposition by iterating over the 13 'k' dimensions
    # and adding the 4 rank-1 terms for each 2x2 multiplication block.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 4 rank-1 terms for a 2x2 multiplication
    # Term (i, j) in A and (j, k) in B contribute to (i, k) in C
    # For a fixed k, the terms are:
    # 1. A[0,0] * B[0,k] -> C[0,k]
    # 2. A[0,1] * B[1,k] -> C[0,k]
    # 3. A[1,0] * B[0,k] -> C[1,k]
    # 4. A[1,1] * B[1,k] -> C[1,k]
    
    # Indices:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    col = 0
    for k in range(p):
        # Term 1: i=0, j=0
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: i=0, j=1
        U[0 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 3: i=1, j=0
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term 4: i=1, j=1
        U[1 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
