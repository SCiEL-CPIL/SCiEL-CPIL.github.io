"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = m * p  # 3 * 9 = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            alpha = j * p + k
            # U: for both i values, set U[i*m + j, alpha] = 1
            for i in range(n):
                U[i * m + j][alpha] = 1
            # V: V[j*p + k, alpha] = 1
            V[j * p + k][alpha] = 1
            # W: for both i values, set W[k*n + i, alpha] = 1
            for i in range(n):
                W[k * n + i][alpha] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
