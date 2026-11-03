"""
Program to find a rank-minimal exact tensor decomposition for multiplying 
a 3 by 5 matrix by a 5 by 6 matrix.

We provide the schoolbook decomposition with rank = n*m*p = 90, which is 
guaranteed to be exact and valid.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: shape (n*m, rank)
    # V: shape (m*p, rank)
    # W: shape (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row i*m + j
                U[i * m + j, column] = 1.0
                # Index into V: row j*p + k
                V[j * p + k, column] = 1.0
                # Index into W: row k*n + i (following W[k*n+i, r] convention)
                W[k * n + i, column] = 1.0
                column += 1
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
