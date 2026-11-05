"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    # Rank-minimal decomposition: rank = m*p = 28
    # For each (j,k) in [0,m) x [0,p):
    #   U[i*m+j, (j,k)] = 1 for all i
    #   V[j*p+k, (j,k)] = 1
    #   W[k*n+i, (j,k)] = 1 for all i
    rank = m * p  # 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * rank for _ in range(p * n)]  # 14 x 28
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
