"""
Matrix multiplication 4x5x6 decomposition.
Uses schoolbook (rank 120) as baseline, with potential for improvement.
"""
import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    # Schoolbook decomposition: rank = n * m * p = 120
    # For each (i, j, k) triple, we have a rank-1 term:
    # U[i*m + j, r] = 1, V[j*p + k, r] = 1, W[k*n + i, r] = 1
    # where r indexes the rank-1 component
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
