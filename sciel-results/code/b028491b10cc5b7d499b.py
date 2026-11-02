"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition for 2x4 x 4x7 matrix multiplication.
    Rank = 56. This is the trivial decomposition where each term
    corresponds to one (i,j,k) triple.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    rank = n * m * p  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (14, 56)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i  (trace convention: W[k*n + i, r])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
