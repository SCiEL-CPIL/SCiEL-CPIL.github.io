"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses the schoolbook decomposition with rank = n*m*p = 60.
    The tensor T_{i,j,k} = A_{i,j} * B_{j,k} is decomposed as sum of rank-1 terms.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    W index follows trace convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
