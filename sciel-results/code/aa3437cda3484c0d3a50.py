"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import expm, logm, solve, inv
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 3*4*5 = 60
    
    # The matrix multiplication tensor T has indices (i, j, k) where:
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i, j, k] = 1 if this term contributes to C[i, k] from A[i, j] * B[j, k]
    
    # We want to decompose T as a sum of rank-1 tensors:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^(n*m), v_r is in R^(m*p), w_r is in R^(p*n)
    
    # For each term (i, j, k) in the schoolbook multiplication, we have:
    # u[i*m + j] = 1, v[j*p + k] = 1, w[k*n + i] = 1
    
    # Let's try to find a lower rank decomposition using numerical optimization.
    # We'll start with a random initialization and try to minimize the rank.
    
    # First, let's try a simple approach: use the standard decomposition but check
    # if we can merge some terms.
    
    # For now, let's return the standard decomposition as a baseline, but with
    # the proper structure.
    
    rank = total_terms
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the required format
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
