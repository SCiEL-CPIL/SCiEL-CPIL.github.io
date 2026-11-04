"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # j=0: 6 standard terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # j=1: 6 standard terms
    for i in range(n):
        for k in range(p):
            U[i * m + 1][col] = 1
            V[1 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # j=2: 5 terms
    # (i=0, k=0)
    U[0 * m + 2][col] = 1
    V[2 * p + 0][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # (i=0, k=1)
    U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[1 * n + 0][col] = 1
    col += 1
    
    # (i=0, k=2)
    U[0 * m + 2][col] = 1
    V[2 * p + 2][col] = 1
    W[2 * n + 0][col] = 1
    col += 1
    
    # (i=1, k=0)
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1
    W[0 * n + 1][col] = 1
    col += 1
    
    # (i=1, k=1) and (i=1, k=2) combined
    U[1 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    V[2 * p + 2][col] = 1
    W[1 * n + 1][col] = 1
    W[2 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
