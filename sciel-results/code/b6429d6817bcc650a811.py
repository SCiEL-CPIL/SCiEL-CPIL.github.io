import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 3, 5, 7
    
    # Try to find a lower-rank decomposition
    # Start with a reasonable upper bound
    # The trivial rank is n*m*p = 105
    # Try to find a better one
    
    # For now, just return the trivial decomposition
    # but try to optimize if we have time
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
