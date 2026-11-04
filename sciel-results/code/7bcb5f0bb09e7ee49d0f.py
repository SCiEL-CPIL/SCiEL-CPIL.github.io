import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor for C = AB is T_ijk = 1 for all i, j, k.
    # This is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # U, V, W can be all ones with shape (n*m, 1), (m*p, 1), (p*n, 1).
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
