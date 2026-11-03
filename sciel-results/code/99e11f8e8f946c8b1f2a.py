import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 4, 8
    dim_U = n * m  # 8
    dim_V = m * p  # 32
    dim_W = p * n  # 16
    
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][idx] = 1
                V[v_idx][idx] = 1
                W[w_idx][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
