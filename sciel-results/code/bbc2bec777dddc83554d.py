"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * p
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        i = c // p
        k = c % p
        
        # W[k*n + i, c] = 1
        W[k * n + i][c] = 1
        
        for j in range(m):
            # U[i*m + j, c] = 1
            U[i * m + j][c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
