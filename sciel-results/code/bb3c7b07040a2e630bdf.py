"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Approach: Vectorized construction of the trivial decomposition.
    # The rank of M_{2,3,3} is 18.
    # We construct the factor matrices U, V, W of shape (n*m, rank), (m*p, rank), (p*n, rank).
    # Each rank-1 term corresponds to a unique (i, j, k) triple.
    
    rank = n * m * p
    
    # Generate indices for all (i, j, k) combinations
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten indices to correspond to the column index in U, V, W (the rank dimension)
    # The order of flattening must be consistent. 
    # Let's flatten in C-order (last index changes fastest).
    # i_vals, j_vals, k_vals are 3D arrays of shape (n, m, p).
    # We want to map each (i, j, k) to a unique column index 'r'.
    # r = i * (m * p) + j * p + k
    
    r_indices = i_vals.ravel() * (m * p) + j_vals.ravel() * p + k_vals.ravel()
    
    # U: shape (n*m, rank). Row index is i*m + j.
    u_rows = i_vals.ravel() * m + j_vals.ravel()
    
    # V: shape (m*p, rank). Row index is j*p + k.
    v_rows = j_vals.ravel() * p + k_vals.ravel()
    
    # W: shape (p*n, rank). Row index is k*n + i.
    # Note: The task specifies W[k*n+i, r].
    w_rows = k_vals.ravel() * n + i_vals.ravel()
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Set the 1s at the appropriate positions
    U[u_rows, r_indices] = 1.0
    V[v_rows, r_indices] = 1.0
    W[w_rows, r_indices] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
