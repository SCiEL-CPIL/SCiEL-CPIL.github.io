import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, the known minimal rank is 30 (schoolbook).
    We construct U, V, W using a Kronecker-product-like structure derived from
    identity matrices to explicitly represent the multiplication as a sum of
    outer products of elementary selection vectors.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We generate the indices for each rank-1 term (i, j, k)
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    
    # Create arrays for indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Calculate row indices for U, V, W
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s at the appropriate positions
    # Each column c corresponds to the c-th (i,j,k) triplet
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    # Convert to lists to match the interface if required, or return arrays
    # The prompt says "factor matrices of shapes...", usually arrays are fine.
    # The current program returns lists of lists. I will return lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
