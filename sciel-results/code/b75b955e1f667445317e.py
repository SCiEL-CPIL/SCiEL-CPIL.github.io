"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Direct schoolbook decomposition for 3x5 * 5x7
    # Rank = n * m * p = 3 * 5 * 7 = 105
    # Each term corresponds to one (i, j, k) triple
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                U[i * m + j][col] = 1
                # V row index: j*p + k
                V[j * p + k][col] = 1
                # W row index: k*n + i
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
