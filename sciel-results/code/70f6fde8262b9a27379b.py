import numpy as np

def construct():
    n, m, p = 2, 5, 6
    # Start with trivial decomposition and try to reduce rank
    # Using a structured approach based on block decomposition
    
    # The trivial rank is n*m*p = 60
    # Let's try to find a lower rank decomposition
    
    # For now, return the trivial decomposition as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
