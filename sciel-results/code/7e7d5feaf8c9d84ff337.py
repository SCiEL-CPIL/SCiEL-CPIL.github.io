import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, we construct the standard (schoolbook) decomposition of rank 30.
    The structure is based on the outer products of the standard basis vectors
    for the input and output spaces, corresponding to the Kronecker product structure
    of the identity operators on the respective index spaces.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Dimensions of the factor matrices
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((dim_u, rank), dtype=int)
    V = np.zeros((dim_v, rank), dtype=int)
    W = np.zeros((dim_w, rank), dtype=int)
    
    # Populate U, V, W based on the tensor structure T[i, j, k]
    # Each term r corresponds to a unique (i, j, k) triplet.
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1 (Trace convention: W index is k*n + i)
    
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
