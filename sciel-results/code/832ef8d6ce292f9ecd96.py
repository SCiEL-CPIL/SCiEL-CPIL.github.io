"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 * p
    rank = 3 * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for the rank 3 decomposition of the 2x3 part
    # u[s][j], v[s][j], w[s][i]
    # Sum_s u[s][j] * v[s][j] * w[s][i] == 1 for all i, j
    u = [
        [1, 1, 1],  # s=0
        [1, 1, 1],  # s=1
        [0, 0, 0]   # s=2
    ]
    v = [
        [1, 1, 1],  # s=0
        [1, 1, 1],  # s=1
        [0, 0, 0]   # s=2
    ]
    w = [
        [0.5, 0.5], # s=0
        [0.5, 0.5], # s=1
        [0, 0]      # s=2
    ]
    
    for k in range(p):
        for s in range(3):
            r = 3 * k + s
            # Fill U: rows i*m + j
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = u[s][j]
            # Fill V: rows j*p + k
            for j in range(m):
                V[j * p + k][r] = v[s][j]
            # Fill W: rows k*n + i
            for i in range(n):
                W[k * n + i][r] = w[s][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
