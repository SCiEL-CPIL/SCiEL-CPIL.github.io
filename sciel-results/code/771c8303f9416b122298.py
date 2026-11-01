"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 2 * 3 = 6
    rank = 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each row i of A (and C)
    for i in range(n):
        # For each column k of A (and row of B)
        for k in range(m):
            # U[i*m + k, col] = 1
            U[i * m + k][col] = 1
            
            # V[k*p + j, col] = 1 for all j
            for j in range(p):
                V[k * p + j][col] = 1
                
            # W[j*n + i, col] = 1 for all j
            for j in range(p):
                W[j * n + i][col] = 1
                
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
