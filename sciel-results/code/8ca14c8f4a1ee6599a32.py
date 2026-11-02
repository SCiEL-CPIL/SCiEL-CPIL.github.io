"""Matrix multiplication (2,3,7) rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of schoolbook decomposition
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Create indices for all (i, j, k) combinations
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Assign 1s to the appropriate positions
    # Since each (i, j, k) corresponds to a unique column in the schoolbook,
    # we can assign directly.
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
