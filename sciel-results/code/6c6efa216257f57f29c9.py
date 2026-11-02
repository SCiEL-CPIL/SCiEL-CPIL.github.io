"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
This version attempts to find a lower rank decomposition for 4x5x6 matrix multiplication.
If no lower rank is found within the time limit, it returns the schoolbook decomposition of rank 120.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Check for simple improvements (none known for 4,5,6 easily)
    # Return schoolbook as it is the most robust valid solution.
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
