"""Matrix multiplication 3,3,7 decomposition using NumPy."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W' containing numpy arrays.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Populate the matrices for the schoolbook algorithm
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
