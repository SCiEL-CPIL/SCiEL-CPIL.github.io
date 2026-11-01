import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses the schoolbook decomposition which has rank n*m*p = 60.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    # Schoolbook Rank
    rank = n * m * p  # 60
    
    # Initialize matrices as numpy arrays for efficiency
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Use vectorized construction for speed
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                
    # Convert to lists for the required interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
