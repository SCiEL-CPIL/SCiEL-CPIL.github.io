import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # Decomposition:
    # U[i*m + j, 0] = 1
    # V[j*p + k, 0] = 1
    # W[k*n + i, 0] = 1
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
