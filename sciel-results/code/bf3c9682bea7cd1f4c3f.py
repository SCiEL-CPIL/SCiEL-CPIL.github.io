import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = 3, 3, 8
    
    # Schoolbook decomposition
    # Rank = n * m * p = 3 * 3 * 8 = 72
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
