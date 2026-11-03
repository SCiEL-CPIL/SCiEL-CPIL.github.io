import numpy as np
import random
import time

def construct():
    n, m, p = 2, 3, 10
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find better decomposition for 2x3x3
    # Target rank 15
    # ... search code ...
    
    return {"U": U_school, "V": V_school, "W": W_school}
