import time
import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (10, rank)
    # W: (p*n, rank) = (10, rank)
    # W index: k*n + i
    
    rank = n * m * p  # trivial decomposition
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][c] = 1.0
                V[v_idx][c] = 1.0
                W[w_idx][c] = 1.0
                c += 1
    
    assert c == rank, f"Expected {rank} terms, got {c}"
    
    return {"U": U, "V": V, "W": W}
