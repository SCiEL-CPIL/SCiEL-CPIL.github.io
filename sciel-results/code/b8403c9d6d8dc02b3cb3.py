"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = 27
    U = [[0] * rank for _ in range(n * m)]  # 9 x 27
    V = [[0] * rank for _ in range(m * p)]  # 15 x 27
    W = [[0] * rank for _ in range(p * n)]  # 15 x 27
    
    r = 0
    # 9 terms: 3x3x3 schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    # 18 terms: 3x3x2 schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
