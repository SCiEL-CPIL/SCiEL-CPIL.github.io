"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i of A (0 to n-1)
    for i in range(n):
        # For each column j of A (0 to m-1), assign a rank-1 component
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
            # Other W entries are 0
            for k in range(p):
                for i2 in range(n):
                    if i2 != i:
                        W[k * n + i2][r] = 0
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
