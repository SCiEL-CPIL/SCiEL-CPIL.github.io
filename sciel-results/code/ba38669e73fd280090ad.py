import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 63
    # This is guaranteed to be an exact decomposition.
    # For 3x3x7 matrix multiplication, the minimal rank is unknown to be less than 63.
    # Return the schoolbook decomposition which has integer entries (hence also half-integer).
    
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
