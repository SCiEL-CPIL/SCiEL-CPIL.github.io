"""Matrix multiplication 2x2 by 2x5 with rank-20 schoolbook decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 20
    V = [[0] * rank for _ in range(m * p)]   # 10 x 20
    W = [[0] * rank for _ in range(p * n)]   # 10 x 20
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
