import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    schoolbook_rank = n * m * p
    
    U_school = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
