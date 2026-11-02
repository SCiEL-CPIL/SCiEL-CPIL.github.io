"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # 5
    # U: shape (n*m, rank) = (10, 5)
    # U[i*m + j, r] = 1 if j == r else 0
    U = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    # V: shape (m*p, rank) = (30, 5)
    # V[j*p + k, r] = 1 if j == r else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W: shape (p*n, rank) = (12, 5)
    # W[k*n + i, r] = 1 for all k, i, r
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
