"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the standard schoolbook tensor decomposition for matrix multiplication.
    Dimensions: n=2, m=3, p=7.
    Rank: n*m*p = 42.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # U shape: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V shape: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W shape: (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the rank-1 components
    # Each component corresponds to a unique (i, j, k) tuple
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i (as per task contract trace convention)
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
