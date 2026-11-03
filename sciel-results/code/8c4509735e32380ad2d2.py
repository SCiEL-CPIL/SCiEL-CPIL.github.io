"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n * m  # 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            # U[i*m + j, c] = 1
            U[i * m + j][c] = 1
            # V[j*p + k, c] = 1 for all k
            for k in range(p):
                V[j * p + k][c] = 1
            # W[k*n + i, c] = 1 for all k
            for k in range(p):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

