"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as per interface requirement if needed, 
    # but numpy arrays are usually accepted if the interface implies array-like.
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # Returning numpy arrays is standard.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
