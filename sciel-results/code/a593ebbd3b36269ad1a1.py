"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 38: 
    # Split m=4 into j=0 (1 dim) and j=1,2,3 (3 dims).
    # Part 1 (j=0): 2 terms (one for i=0, one for i=1) using full k range.
    # Part 2 (j=1,2,3): Schoolbook 2*3*6 = 36 terms.
    # Total Rank = 2 + 36 = 38.
    
    rank = 38
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    # A[0, 0] * B[0, k] -> C[0, k]
    # A[1, 0] * B[0, k] -> C[1, k]
    
    # Term for i=0, j=0
    U[0 * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
    col += 1
    
    # Term for i=1, j=0
    U[1 * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
    col += 1
    
    # Part 2: j in {1, 2, 3}
    # Schoolbook: A[i, j] * B[j, k] -> C[i, k]
    for i in range(n):
        for j in range(1, m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
