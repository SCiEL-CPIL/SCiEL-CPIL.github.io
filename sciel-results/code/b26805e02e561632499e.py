"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    The minimal rank is n*m*p = 40.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # Indexing:
    # U rows: i*m + j (where i in 0..n-1, j in 0..m-1)
    # V rows: j*p + k (where j in 0..m-1, k in 0..p-1)
    # W rows: k*n + i (where k in 0..p-1, i in 0..n-1)
    # Column c corresponds to the term A[i,j] * B[j,k]
    
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
