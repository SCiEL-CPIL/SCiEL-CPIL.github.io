"""
Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition.

The trivial decomposition has rank n*m*p = 60.
This program constructs that decomposition efficiently using pure Python lists.
"""

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = n * m * p  # 60
    
    # U has shape (n*m, R) = (6, 60)
    # V has shape (m*p, R) = (30, 60)
    # W has shape (p*n, R) = (20, 60)
    
    # Initialize all zeros
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # For each (i, j, k) in range(n) x range(m) x range(p):
    #   col = (i*m + j)*p + k
    #   U[i*m + j, col] = 1
    #   V[j*p + k, col] = 1
    #   W[k*n + i, col] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                rowU = i * m + j
                rowV = j * p + k
                rowW = k * n + i
                U[rowU][col] = 1
                V[rowV][col] = 1
                W[rowW][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
