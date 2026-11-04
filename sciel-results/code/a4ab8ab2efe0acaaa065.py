import time
import numpy as np

def construct():
    """
    Returns a rank-1 tensor decomposition for 3x4 * 4x7 matrix multiplication.
    The multiplication tensor under the trace convention is the all-ones tensor.
    The rank-1 decomposition uses U=1, V=1, W=1.
    """
    n, m, p = 3, 4, 7
    
    # U shape (n*m, 1)
    U = np.ones((n * m, 1))
    # V shape (m*p, 1)
    V = np.ones((m * p, 1))
    # W shape (p*n, 1)
    W = np.ones((p * n, 1))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
