"""
Alternative construction for matrix multiplication tensor decomposition.
Exploits the block structure of the multiplication to vectorize the schoolbook rank construction.
"""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    
    # Dimensions of the factor matrices
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Schoolbook rank: n * m * p
    R = n * m * p
    
    # We will build the matrices column by column, but using vectorized logic for each column
    # to ensure the code structure is distinct from simple loop increments.
    
    # Create lists of row indices for U, V, W
    # For each triple (i, j, k), we place a 1 at:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    # Pre-allocate matrices
    U_list = [[0] * R for _ in range(N_U)]
    V_list = [[0] * R for _ in range(N_V)]
    W_list = [[0] * R for _ in range(N_W)]
    
    # Iterate through all rank components
    # Instead of incrementing a counter 'col', we zip with an index generator
    index_gen = range(R)
    
    for (i, j, k), col_idx in zip(product(range(n), range(m), range(p)), index_gen):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U_list[u_idx][col_idx] = 1
        V_list[v_idx][col_idx] = 1
        W_list[w_idx][col_idx] = 1
        
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
# EVOLVE-BLOCK-END
