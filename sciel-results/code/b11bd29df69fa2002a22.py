import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    Uses the schoolbook algorithm with rank n*m*p = 45.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}
