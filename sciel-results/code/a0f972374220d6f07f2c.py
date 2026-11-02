"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the matrix multiplication tensor decomposition.
    The decomposition is exact: sum_{c} U[:, c] V[:, c] W[:, c] == T
    where T is the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
