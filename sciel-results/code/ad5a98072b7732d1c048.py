"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 27, the theoretical lower bound
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            col = j * p + k
            # V[j*p+k, col] = 1 (identity-like)
            V[j * p + k][col] = 1
            # U[i*m+j, col] = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            # W[k*n+i, col] = 1 for all i
            for i in range(n):
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
