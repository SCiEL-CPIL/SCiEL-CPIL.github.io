import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-125 exact tensor decomposition for 5x5 x 5x5 matrix multiplication.
    Uses vectorized NumPy construction for the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create indices
    # U[i*m + j, r] = 1 if r corresponds to (i,j,k)
    # V[j*p + k, r] = 1 if r corresponds to (i,j,k)
    # W[k*n + i, r] = 1 if r corresponds to (i,j,k)
    
    # We can create the matrices by iterating over the rank dimension
    # Or using broadcasting
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate all (i, j, k) tuples
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # The rank index r is just the position in the flattened array
    r_idx = np.arange(rank)
    
    U[u_idx, r_idx] = 1
    V[v_idx, r_idx] = 1
    W[w_idx, r_idx] = 1
    
    # Convert to lists to match seed format if necessary, 
    # but NumPy arrays are usually fine. 
    # The seed returns lists. Let's return lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
