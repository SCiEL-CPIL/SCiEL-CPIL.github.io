"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = n * m  # 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for a in range(n):
        for b in range(m):
            r = a * m + b
            U[a * m + b][r] = 1
            for j in range(m):
                for k in range(p):
                    V[j * p + k][r] = 1
            for k in range(p):
                for i in range(n):
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
