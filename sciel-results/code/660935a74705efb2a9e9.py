"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2
    
    # U shape (n*m, rank) = (4, 2)
    # U[i*m + j, c]
    U = [[0] * rank for _ in range(n * m)]
    # V shape (m*p, rank) = (22, 2)
    # V[j*p + k, c]
    V = [[0] * rank for _ in range(m * p)]
    # W shape (p*n, rank) = (22, 2)
    # W[k*n + i, c]
    W = [[0] * rank for _ in range(p * n)]
    
    # Component 0: handles i=0, k=0
    for j in range(m):
        U[0 * m + j][0] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    W[0 * n + 0][0] = 1  # k=0, i=0
    
    # Component 1: handles i=1, k=1
    for j in range(m):
        U[1 * m + j][1] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][1] = 1
    W[1 * n + 1][1] = 1  # k=1, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
