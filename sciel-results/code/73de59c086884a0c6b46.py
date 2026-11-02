"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = 2 * p for n=2, m=2
    rank = 2 * p
    U = [[0] * rank for _ in range(n * m)]  # (4, rank)
    V = [[0] * rank for _ in range(m * p)]  # (26, rank)
    W = [[0] * rank for _ in range(p * n)]  # (26, rank)
    
    for k in range(p):
        # Column 2k: handles j=0
        c0 = 2 * k
        # U[i*m + j, c] for j=0, i=0,1 -> indices 0, 2
        U[0][c0] = 1   # i=0, j=0
        U[2][c0] = 1   # i=1, j=0
        # V[j*p + k, c] for j=0 -> index k
        V[k][c0] = 1
        # W[k*n + i, c] for i=0,1 -> indices 2k, 2k+1
        W[2 * k][c0] = 1
        W[2 * k + 1][c0] = 1
        
        # Column 2k+1: handles j=1
        c1 = 2 * k + 1
        # U[i*m + j, c] for j=1, i=0,1 -> indices 1, 3
        U[1][c1] = 1   # i=0, j=1
        U[3][c1] = 1   # i=1, j=1
        # V[j*p + k, c] for j=1 -> index p + k
        V[p + k][c1] = 1
        # W[k*n + i, c] for i=0,1 -> indices 2k, 2k+1
        W[2 * k][c1] = 1
        W[2 * k + 1][c1] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
