"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Return as lists to match the interface requirement if strict types are needed, 
    # but numpy arrays are usually accepted or convertible. 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # Returning lists of lists is safer for JSON serialization if the evaluator uses that.
    return {
        "U": U.tolist(), 
        "V": V.tolist(), 
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
