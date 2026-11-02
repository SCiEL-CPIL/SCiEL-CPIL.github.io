"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, r] = 1 if j == r
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
            
    # V[j*p + k, r] = 1 if j == r
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
            
    # W[k*n + i, r] = 1
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
