"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a rank-minimal (or near-minimal) exact tensor decomposition 
    for matrix multiplication of dimensions 3x5 by 5x5.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The decomposition satisfies:
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 
    for all valid i, j, k corresponding to the multiplication tensor.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Fill in the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (trace convention)
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists as required by the interface if needed, 
    # but the prompt says "integer/half-integer entries". 
    # The interface example used lists. Let's check if numpy arrays are accepted.
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices"
    # Usually, numpy arrays are fine, but to be safe with "integer/half-integer", 
    # and the seed used lists, I will return lists of lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
