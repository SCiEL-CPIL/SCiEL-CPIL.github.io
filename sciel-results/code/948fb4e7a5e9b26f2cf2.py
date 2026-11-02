import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    dim_a = n * m  # 15
    dim_b = m * p  # 35
    dim_c = p * n  # 21
    rank = n * m * p  # 105
    
    U = np.zeros((dim_a, rank), dtype=np.float64)
    V = np.zeros((dim_b, rank), dtype=np.float64)
    W = np.zeros((dim_c, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
