import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_solution = (U_sb, V_sb, W_sb)
    
    # Convert best solution to list of lists
    U_list = best_solution[0].tolist()
    V_list = best_solution[1].tolist()
    W_list = best_solution[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
