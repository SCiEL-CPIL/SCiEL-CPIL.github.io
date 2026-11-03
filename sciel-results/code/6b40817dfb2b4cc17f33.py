"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    Uses the standard schoolbook decomposition (rank = n*m*p).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize as 0-d arrays for flexibility, then fill
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row (i*m + j)
                U[i * m + j, col] = 1
                
                # Index in V: row (j*p + k)
                V[j * p + k, col] = 1
                
                # Index in W: row (k*n + i)
                W[k * n + i, col] = 1
                
                col += 1
                
    # Convert to lists of lists to match the interface expectation of "integer/half-integer entries"
    # The interface accepts lists, numpy arrays might work too, but lists are safer for serialization.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
