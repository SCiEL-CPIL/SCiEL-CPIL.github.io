import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to use Winograd 23-rank for 3x3x3
    # Split p=8 into 3+3+2
    # Rank 23 + 23 + 18 (schoolbook for 3x3x2) = 64
    # We need 23-rank matrices for 3x3x3
    
    # I will use a known set of 23-rank matrices if available
    # Otherwise, fall back to schoolbook
    
    # For now, I will return the schoolbook to ensure validity.
    # If I can find the 23-rank matrices, I will use them.
    
    return {"U": U_school, "V": V_school, "W": W_school}
