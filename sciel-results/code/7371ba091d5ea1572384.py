"""
Program for Matrix Multiplication 3,5,7 rank minimization.
Uses vectorized construction via NumPy for the schoolbook decomposition.
"""

import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros using NumPy
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j, col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k, col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists to match the expected interface if strict types are required,
    # though the problem statement says "integer/half-integer entries", 
    # and numpy int is usually accepted or can be converted.
    # The original returned lists of lists. Let's return lists of lists to be safe.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
