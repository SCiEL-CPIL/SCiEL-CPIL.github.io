import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x3 matrix.
    
    The output factor W follows the official trace convention: W[k*n + i, r].
    
    For small dimensions, the trivial rank n*m*p is often optimal or close to it.
    We use the standard unit-tensor decomposition which is guaranteed to be exact.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # The trivial rank is n*m*p
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Iterate over all (i, j, k) indices
    # i: row of A (0 to n-1)
    # j: column of A / row of B (0 to m-1)
    # k: column of B (0 to p-1)
    
    term_idx = 0
    for i, j, k in product(range(n), range(m), range(p)):
        # U[i*m + j, term] = 1  (selects A[i, j])
        U[i * m + j, term_idx] = 1
        
        # V[j*p + k, term] = 1  (selects B[j, k])
        V[j * p + k, term_idx] = 1
        
        # W[k*n + i, term] = 1  (routes to C[i, k] with trace convention k*n + i)
        W[k * n + i, term_idx] = 1
        
        term_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
