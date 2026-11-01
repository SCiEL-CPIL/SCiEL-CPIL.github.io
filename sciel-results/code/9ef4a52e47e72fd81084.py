# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Rank = p (since M_{2,2,1} has rank 1)
    rank = p
    U = [[0.0] * rank for _ in range(n * m)]  # shape (4, rank)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (18, rank)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (18, rank)
    
    # For each k, we use one rank-1 term
    for k in range(p):
        r = k
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
