import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    For dimensions n x m and m x p, the rank is n*m*p.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U[i*m + j, c] = 1 if c corresponds to (i, j, k)
    # V[j*p + k, c] = 1 if c corresponds to (i, j, k)
    # W[k*n + i, c] = 1 if c corresponds to (i, j, k)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                U[i * m + j][column] = 1
                # Index in V: row j*p + k
                V[j * p + k][column] = 1
                # Index in W: row k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
