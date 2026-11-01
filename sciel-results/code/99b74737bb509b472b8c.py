import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 2 * m  # 2 * 3 = 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each j in 0..m-1, we need to satisfy T[i, j, k] = delta_{j,k}
    # This means for a fixed j, T[i, j, j] = 1 for all i, and 0 otherwise.
    # We can decompose this using 2 rank-1 tensors for each j.
    # One for i=0, one for i=1.
    
    for j in range(m):
        # Term for i=0
        c0 = 2 * j
        U[0 * m + j][c0] = 1
        V[j * p + j][c0] = 1
        W[j * n + 0][c0] = 1
        
        # Term for i=1
        c1 = 2 * j + 1
        U[1 * m + j][c1] = 1
        V[j * p + j][c1] = 1
        W[j * n + 1][c1] = 1
        
    return {"U": U, "V": V, "W": W}
