import numpy as np
import time

def construct():
    """
    Constructs the tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    Uses the schoolbook algorithm which has rank 54.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U has shape (2*3, 54) = (6, 54)
    V has shape (3*9, 54) = (27, 54)
    W has shape (9*2, 54) = (18, 54)
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 2, 3, 9
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
                r += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
