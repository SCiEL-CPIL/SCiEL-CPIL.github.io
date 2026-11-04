"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook tensor decomposition for C = A @ B
    # A is (n x m) = 4x4, B is (m x p) = 4x6, C is (n x p) = 4x6
    # Rank = n * m * p = 96
    # U: (n*m) x rank = 16 x 96
    # V: (m*p) x rank = 24 x 96
    # W: (p*n) x rank = 24 x 96, indexed as W[k*n + i, r]
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
