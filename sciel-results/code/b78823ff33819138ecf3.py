import numpy as np
import time
import sys

def construct():
    """
    Returns the rank-27 schoolbook decomposition for 3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n, m, p = 3, 3, 3
    
    # Generate schoolbook 27 terms
    U_school = np.zeros((9, 27))
    V_school = np.zeros((9, 27))
    W_school = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_school[i * 3 + j, col] = 1
                V_school[j * 3 + k, col] = 1
                W_school[k * 3 + i, col] = 1
                col += 1

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
