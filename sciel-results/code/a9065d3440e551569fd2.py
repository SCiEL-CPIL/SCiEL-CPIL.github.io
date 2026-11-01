import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Decomposition into p independent (n, m, 1) tensors.
    # Rank of (2, 3, 1) is 6. Total rank is 9 * 6 = 54.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx, (i, j, k) in enumerate(
        (i, j, k) for i in range(n) for j in range(m) for k in range(p)
    ):
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
        
    return {"U": U, "V": V, "W": W}
