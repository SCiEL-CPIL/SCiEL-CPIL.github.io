"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    
    The problem decomposes into 15 independent 2x2 * 2x1 multiplications.
    The rank of 2x2 * 2x1 is 4.
    Total rank = 15 * 4 = 60.
    
    We construct the decomposition using NumPy for clarity and speed.
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    rank = n * m * p       # 60
    
    # Initialize factor matrices as NumPy arrays for easier indexing
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill in the decomposition
    # Index c corresponds to tuple (i, j, k)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    
    # Convert back to lists if required, but dict of arrays is usually accepted.
    # The task says "factor matrices of shapes...". Lists of lists are safe.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
