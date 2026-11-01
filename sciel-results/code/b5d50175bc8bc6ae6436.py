"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for C = A @ B where A is n x m and B is m x p
    # is defined such that T[i, j, k] = 1 for all valid i, j, k.
    # This is because the term A[i, j] * B[j, k] contributes to C[i, k] for all i, j, k.
    # Therefore, the tensor is the all-ones tensor of size n x m x p.
    # The rank of the all-ones tensor is 1, as it can be decomposed into a single
    # rank-1 term: T[i, j, k] = 1 * 1 * 1.
    
    rank = 1
    
    # U has shape (n*m, rank)
    # Indexing: U[i*m + j, r]
    U = [[0] * rank for _ in range(n * m)]
    # V has shape (m*p, rank)
    # Indexing: V[j*p + k, r]
    V = [[0] * rank for _ in range(m * p)]
    # W has shape (p*n, rank)
    # Indexing: W[k*n + i, r]
    W = [[0] * rank for _ in range(p * n)]
    
    # Set all entries to 1 for the single rank component r=0
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
            
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
            
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
