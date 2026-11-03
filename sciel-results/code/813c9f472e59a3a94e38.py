import numpy as np
import time

DIMENSIONS = [3, 4, 7]
DEADLINE_SECONDS = 140  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n * m * p = 3 * 4 * 7 = 84
    U_rows = []
    V_rows = []
    W_rows = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_rows.append((i, j))
                V_rows.append((j, k))
                W_rows.append((k, i))
    
    rank = len(U_rows)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        i, j = U_rows[col]
        j2, k = V_rows[col]
        k2, i2 = W_rows[col]
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    result = {"U": U, "V": V, "W": W}
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        return result
    
    return result
