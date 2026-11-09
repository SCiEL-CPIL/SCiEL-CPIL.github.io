import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U index: i*m + j
            U[i * m + j][col] = 1
            
            # V index: j*p + k for all k
            for k in range(p):
                V[j * p + k][col] = 1
                
            # W index: k*n + i for all k
            for k in range(p):
                W[k * n + i][col] = 1
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
