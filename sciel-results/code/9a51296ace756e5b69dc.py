"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication using the schoolbook method.
    This approach is guaranteed to be exact and valid.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all terms
    # Term r corresponds to (i, j, k) where r = i * m * p + j * p + k
    i_indices, j_indices, k_indices = np.indices((n, m, p))
    
    # Flatten indices
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()
    k_flat = k_indices.flatten()
    
    # Calculate linear indices for U, V, W
    u_indices = i_flat * m + j_flat
    v_indices = j_flat * p + k_flat
    w_indices = k_flat * n + i_flat
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Set values
    U[u_indices, np.arange(rank)] = 1
    V[v_indices, np.arange(rank)] = 1
    W[w_indices, np.arange(rank)] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
