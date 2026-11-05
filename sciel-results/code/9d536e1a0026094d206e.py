import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split p into p1=4 and p2=1
    p1 = 4
    p2 = 1
    
    rank1 = n * m * p1  # 16
    rank2 = n * m * p2  # 16
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4x4
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 4x4x1
    # B is 4x5. We are taking the last column (index 4)
    # V index is j*p + k. k=4.
    # W index is k*n + i. k=4.
    for i in range(n):
        for j in range(m):
            k = p1
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
