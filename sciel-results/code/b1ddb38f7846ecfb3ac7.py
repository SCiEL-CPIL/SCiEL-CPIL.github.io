"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 blocks of Strassen (rank 7)
    rank = 28
    
    # Strassen matrices for 2x2 multiplication
    # U rows: a, b, c, d
    # V rows: e, f, g, h
    # W rows: C00, C10, C01, C11 (index k*2+i)
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ]
    V_strassen = [
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, -1, 1, 1, 0]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(4):
        for r in range(7):
            col = j * 7 + r
            # U is shared
            for i in range(2):
                for l in range(2):
                    U[i * m + l][col] = U_strassen[i * 2 + l][r]
            
            # V is block diagonal
            for l in range(2):
                for k in range(2):
                    # k_global = 2*j + k
                    k_global = 2 * j + k
                    V[l * p + k_global][col] = V_strassen[l * 2 + k][r]
            
            # W is block diagonal
            for k in range(2):
                for i in range(2):
                    # k_global = 2*j + k
                    k_global = 2 * j + k
                    W[k_global * n + i][col] = W_strassen[k * 2 + i][r]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
