DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # Since each 2x1xp block has rank 1, and there are m such blocks
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        col = j
        # U rows: i*m + j for i in 0..n-1
        for i in range(n):
            U[i * m + j][col] = 1
        # V rows: j*p + k for k in 0..p-1
        for k in range(p):
            V[j * p + k][col] = 1
        # W rows: k*n + i for k in 0..p-1, i in 0..n-1
        for k in range(p):
            for i in range(n):
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
