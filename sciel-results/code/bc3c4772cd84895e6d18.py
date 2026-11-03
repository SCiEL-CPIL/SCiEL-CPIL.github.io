import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for matrix multiplication.
    The decomposition is the standard schoolbook algorithm.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the seed program's interface, 
    # though numpy arrays are also acceptable if the evaluator handles them.
    # The seed program used lists of lists. I will stick to that to be safe.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for the schoolbook decomposition
    # Each term in the sum corresponds to a specific (i, j, k) index
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by [i*m + j, col]
                U[i * m + j][col] = 1
                # V is indexed by [j*p + k, col]
                V[j * p + k][col] = 1
                # W is indexed by [k*n + i, col]
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
