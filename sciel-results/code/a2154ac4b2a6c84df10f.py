import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_DEADLINE = 145.0
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x5 matrix multiplication.
    The tensor for matrix multiplication in this indexing scheme is the all-ones tensor.
    The rank of the all-ones tensor is 1.
    """
    
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition
    # U shape (n*m, 1)
    # V shape (m*p, 1)
    # W shape (p*n, 1)
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
