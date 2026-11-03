"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1  # U[i*m+j, c] = 1 if c == j
        for k in range(p):
            for j in range(m):
                V[j * p + k][j] = 1  # V[j*p+k, c] = 1 if c == j
            W[k * n + i][0] = 1
            W[k * n + i][1] = 1
            W[k * n + i][2] = 1  # W[k*n+i, c] = 1 for all c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
