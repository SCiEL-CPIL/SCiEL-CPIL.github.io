"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the exact tensor decomposition for 2x5x6 matrix multiplication.
    Uses the standard schoolbook decomposition of rank n*m*p.
    Implemented using NumPy for vectorized index generation.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all (i, j, k) triples
    # i in [0, n), j in [0, m), k in [0, p)
    i_vals, j_vals, k_vals = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    
    # Flatten to 1D arrays
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate flattened indices for U, V, W
    # U index: i*m + j
    u_indices = i_flat * m + j_flat
    # V index: j*p + k
    v_indices = j_flat * p + k_flat
    # W index: k*n + i
    w_indices = k_flat * n + i_flat
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s at the appropriate positions
    # The column index in the decomposition is the index of the term in the sum
    # Since we iterate through all n*m*p terms, the term index is just the range(rank)
    col_indices = np.arange(rank)
    
    U[u_indices, col_indices] = 1
    V[v_indices, col_indices] = 1
    W[w_indices, col_indices] = 1
    
    # Convert to lists of lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
