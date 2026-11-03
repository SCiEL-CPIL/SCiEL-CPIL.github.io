import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Build schoolbook
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Target tensor T[i, j, k]
    # Based on the seed, T[i, j, k] = 1 for all i, j, k?
    # No, the seed produces 1 for each (i,j,k) in the sum.
    # So T is the all-ones tensor?
    # If T is all-ones, rank 1 works.
    # So T must be something else.
    
    # Let's assume the standard MM tensor.
    # I will return the schoolbook.
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
