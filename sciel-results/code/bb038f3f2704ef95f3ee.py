"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # 25s safety margin before 150s limit
    
    # Schoolbook decomposition (rank = n*m*p = 63)
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_schoolbook = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_schoolbook = np.zeros((p * n, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    # Return as lists of integers to ensure exactness
    return {
        "U": U_schoolbook.tolist(),
        "V": V_schoolbook.tolist(),
        "W": W_schoolbook.tolist()
    }
