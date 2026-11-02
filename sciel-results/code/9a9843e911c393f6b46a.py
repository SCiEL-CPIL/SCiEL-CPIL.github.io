import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # Schoolbook solution (Rank 56)
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school), dtype=np.float64)
    V_school = np.zeros((dim_v, rank_school), dtype=np.float64)
    W_school = np.zeros((dim_w, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
