"""Program for matrix multiplication 2,3,7 rank decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42, trivial decomposition
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                # U index: i * m + j
                U[i * m + j][r] = 1
                # V index: j * p + k
                V[j * p + k][r] = 1
                # W index: k * n + i
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
