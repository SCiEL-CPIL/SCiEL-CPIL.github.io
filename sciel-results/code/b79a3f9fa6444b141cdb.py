"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * m
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            
            # U[i*m + j, r] = 1
            U[i * m + j][r] = 1
            
            # V[j*p + k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
            
            # W[k*n + i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
