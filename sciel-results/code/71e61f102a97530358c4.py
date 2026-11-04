import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook), dtype=np.int8)
    V_schoolbook = np.zeros((m * p, rank_schoolbook), dtype=np.int8)
    W_schoolbook = np.zeros((p * n, rank_schoolbook), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    # Convert to list format for the return value
    return {
        "U": U_schoolbook.tolist(),
        "V": V_schoolbook.tolist(),
        "W": W_schoolbook.tolist()
    }
