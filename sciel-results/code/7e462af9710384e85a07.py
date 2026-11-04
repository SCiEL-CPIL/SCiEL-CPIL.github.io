"""Matrix multiplication 2x3 * 3x9 - rank-minimal exact tensor decomposition."""

import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook (rank n*m*p = 54) exact tensor decomposition
    for multiplying a 2x3 matrix by a 3x9 matrix.
    
    The tensor T is defined by:
        T[i, j, k] = 1  (the coefficient of A[i,j] * B[j,k] in C[i,k])
    
    Decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    where:
        U has shape (n*m, rank)  - indices (i, j)
        V has shape (m*p, rank)  - indices (j, k)
        W has shape (p*n, rank)  - indices (k, i)  [trace convention W[k*n+i, r]]
    
    For each term r corresponding to (i, j, k):
        U[i*m + j, r] = 1
        V[j*p + k, r] = 1
        W[k*n + i, r] = 1
    
    This gives an exact decomposition with rank = n*m*p = 54.
    """
    deadline = time.time() + 140  # Safety margin before 150s wall-clock limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
