import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    The bilinear rank is n*m*p = 60.
    This implementation uses a vectorized approach to construct the U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We create the indices for the non-zero entries
    # Each column 'r' in U, V, W corresponds to a unique (i, j, k) tuple
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate all combinations of i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    # r is the index in the rank dimension
    
    # Using meshgrid to generate indices
    i_indices, j_indices, k_indices = np.indices((n, m, p))
    i_indices = i_indices.flatten()
    j_indices = j_indices.flatten()
    k_indices = k_indices.flatten()
    
    # r indices are just 0 to rank-1
    r_indices = np.arange(rank)
    
    # Compute linear indices for U, V, W
    u_indices = i_indices * m + j_indices
    v_indices = j_indices * p + k_indices
    w_indices = k_indices * n + i_indices
    
    # Set the values to 1
    U[u_indices, r_indices] = 1
    V[v_indices, r_indices] = 1
    W[w_indices, r_indices] = 1
    
    # Convert to lists as required by the interface (though numpy arrays might be accepted, 
    # the seed returned lists, so we stick to lists to be safe or check if numpy is allowed in return)
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # Usually numpy arrays are fine, but the seed used lists. 
    # "Keep the function names and signatures ... return interface EXACTLY as in the current program"
    # Current program returns {"U": U, "V": V, "W": W} where U, V, W are lists of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
