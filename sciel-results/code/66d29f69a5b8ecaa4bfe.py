"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Trivial (schoolbook) decomposition for 3x5x5 matrix multiplication.
    
    Rank is n*m*p = 3*5*5 = 75.
    Each term corresponds to a single (i, j, k) index triple.
    U[i*m + j, c] = 1 if c maps to (i,j,k)
    V[j*p + k, c] = 1 if c maps to (i,j,k)
    W[k*n + i, c] = 1 if c maps to (i,j,k)
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n * m * p  # 75
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map (i, j, k) to column c
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
