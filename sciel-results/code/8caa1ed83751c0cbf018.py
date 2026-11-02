import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 105)
    # This is the standard exact decomposition for matrix multiplication.
    # T[i, j, k] = 1 corresponds to the term A[i, j] * B[j, k] contributing to C[i, k].
    # The tensor is decomposed into n*m*p rank-1 terms.
    
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i, col j of A -> index i*m + j
                U[i * m + j][col] = 1
                
                # V index: row j, col k of B -> index j*p + k
                V[j * p + k][col] = 1
                
                # W index: row k, col i of C (transposed convention W[k*n+i]) -> index k*n + i
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
