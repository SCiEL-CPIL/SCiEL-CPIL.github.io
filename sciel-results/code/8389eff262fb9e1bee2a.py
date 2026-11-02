"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    for i in range(n):
        for j in range(m):
            # r corresponds to j
            U[i * m + j][j] = 1
            
    for j in range(m):
        for k in range(p):
            # r corresponds to j
            V[j * p + k][j] = 1
            
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
