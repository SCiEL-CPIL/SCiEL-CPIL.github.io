import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication M_{3,3,5}.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to stay safely under 120s/150s

    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline:
                    # Return current best (which is partial, but schoolbook is built sequentially)
                    # Actually, if we break, the decomposition is invalid.
                    # So we must ensure we finish. 45 iterations is fast.
                    pass
                
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
