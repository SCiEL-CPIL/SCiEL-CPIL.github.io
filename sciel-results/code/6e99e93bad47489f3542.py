"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(p):
        for i in range(n):
            # W index is j * n + i
            W[j * n + i][col] = 1
            
            # U index is i * m + k. Set to 1 for all k if row is i
            for k in range(m):
                U[i * m + k][col] = 1
                
            # V index is k * p + j. Set to 1 for all k
            for k in range(m):
                V[k * p + j][col] = 1
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
