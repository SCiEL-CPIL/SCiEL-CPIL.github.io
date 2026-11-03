"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 4 * 7 = 28
    
    # U: (n*m, rank) = (12, 28)
    # U[i*m + j, j*p + k] = 1
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank) = (28, 28) - identity
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank) = (21, 28)
    # W[k*n + i, j*p + k] = 1
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            V[j * p + k][c] = 1  # V is identity
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
