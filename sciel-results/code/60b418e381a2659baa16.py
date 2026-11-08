"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8: for each of the m=4 middle indices, use rank-2 decomposition of T_{2,1,6}
    # T_{2,1,6} has rank 2 since it's essentially a 2x6 all-ones matrix (rank 2)
    rank = m * 2  # 4 * 2 = 8
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 8)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 8)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 8)
    
    col = 0
    for j in range(m):  # j = 0, 1, 2, 3
        # Two rank-1 terms for the j-th middle index
        # Term 1: contributes to i=0
        # U: A[0, j] coefficient 1
        U[0 * m + j][col] = 1
        # V: B[j, k] for all k, coefficient 1
        for k in range(p):
            V[j * p + k][col] = 1
        # W: C[k, 0] for all k, coefficient 1
        for k in range(p):
            W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: contributes to i=1
        # U: A[1, j] coefficient 1
        U[1 * m + j][col] = 1
        # V: B[j, k] for all k, coefficient 1
        for k in range(p):
            V[j * p + k][col] = 1
        # W: C[k, 1] for all k, coefficient 1
        for k in range(p):
            W[k * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
