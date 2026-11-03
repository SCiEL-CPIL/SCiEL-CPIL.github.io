"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 2 * 5 = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for a in range(m):
            # U[i*m + j, col] = 1 if j == a, else 0
            for j in range(m):
                if j == a:
                    U[i * m + j][col] = 1
            # V[j*p + k, col] = 1 if j == a, else 0 (independent of k)
            for j in range(m):
                if j == a:
                    for k in range(p):
                        V[j * p + k][col] = 1
            # W[k*n + i, col] = 1 for all k, i
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
