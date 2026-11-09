import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    n, m, p = 2, 3, 4
    
    # First, create schoolbook rank 24
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    
    # Try to find rank 22
    # ... optimization code ...
    
    # Return best result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
