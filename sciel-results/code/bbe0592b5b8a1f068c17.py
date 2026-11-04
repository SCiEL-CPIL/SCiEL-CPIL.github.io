import numpy as np
import time

def construct():
    n, m, p = 2, 5, 5
    
    # The tensor for matrix multiplication of n x m and m x p matrices
    # is the all-ones tensor T[i, j, k] = 1.
    # This tensor has rank 1, given by:
    # U[i*m + j, 0] = 1
    # V[j*p + k, 0] = 1
    # W[k*n + i, 0] = 1
    
    rank = 1
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
