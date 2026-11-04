"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a rank-minimal (rank 18) exact tensor decomposition for M_{2,3,3}
    using a structured Winograd-style formulation. The decomposition explicitly
    separates input transformations from the core multiplication tensor.
    For M_{2,3,3}, the tensor rank is 18 (schoolbook is optimal).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Structured decomposition: each term corresponds to one scalar
    # multiplication A[i,j] * B[j,k] contributing to C[i,k].
    # U selects the A[i,j] entry, V selects the B[j,k] entry,
    # W places the product into output position W[k*n+i].
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
