import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1.0
                # V index: j*p + k
                V[j * p + k, col] = 1.0
                # W index: k*n + i (following the trace convention W[k*n+i, r])
                W[k * n + i, col] = 1.0
                col += 1
                
    # Return as lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
