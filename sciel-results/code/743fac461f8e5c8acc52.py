"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p
    # U: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        r1 = 2 * k
        r2 = 2 * k + 1
        
        # Term 1: handles j=0
        # U indices: i=0, j=0 -> 0; i=1, j=0 -> 1
        U[0 * m + 0][r1] = 1
        U[1 * m + 0][r1] = 1
        # V indices: j=0, k=k -> 0*p + k = k
        V[0 * p + k][r1] = 1
        # W indices: k=k, i=0 -> k*n + 0; k=k, i=1 -> k*n + 1
        W[k * n + 0][r1] = 1
        W[k * n + 1][r1] = 1
        
        # Term 2: handles j=1
        # U indices: i=0, j=1 -> 1; i=1, j=1 -> 3
        U[0 * m + 1][r2] = 1
        U[1 * m + 1][r2] = 1
        # V indices: j=1, k=k -> 1*p + k
        V[1 * p + k][r2] = 1
        # W indices: k=k, i=0 -> k*n + 0; k=k, i=1 -> k*n + 1
        W[k * n + 0][r2] = 1
        W[k * n + 1][r2] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
