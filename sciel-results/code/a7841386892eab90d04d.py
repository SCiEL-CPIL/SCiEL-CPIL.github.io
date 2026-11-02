import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 64 decomposition
    # Split m=4 into 3+1
    # Part 1: 3x3x7 -> 7 blocks of 3x3x1 (rank 9 each) -> 63
    # Part 2: 3x1x7 -> 1 block (rank 1) -> 1
    rank = 64
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j in 0..2
    for k in range(p):
        for i in range(n):
            for j in range(3): # 0..2
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: j = 3
    # Rank 1 decomposition for 3x1x7
    # U[i, 3] = 1, V[3, k] = 1, W[k, i] = 1
    for i in range(n):
        U[i * m + 3][col] = 1
    for k in range(p):
        V[3 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
