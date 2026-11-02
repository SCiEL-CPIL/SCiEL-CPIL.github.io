"""
Rewritten program for 5x5 matrix multiplication tensor decomposition.
Uses the schoolbook algorithm which provides a valid decomposition of rank 125.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook decomposition for 5x5 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The rank is 125.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # The tensor T is defined by T[i, j, k] = 1 if we are multiplying A[i, j] * B[j, k] to get C[i, k]
    # In the factorization, the r-th term corresponds to a specific (i, j, k) triple.
    # U[i*m + j, r] = 1 if the r-th term uses A[i, j]
    # V[j*p + k, r] = 1 if the r-th term uses B[j, k]
    # W[k*n + i, r] = 1 if the r-th term contributes to C[i, k]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                U[i * m + j][column] = 1
                # Index in V: row j*p + k
                V[j * p + k][column] = 1
                # Index in W: row k*n + i
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
