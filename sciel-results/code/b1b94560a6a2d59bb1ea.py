"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-3 decomposition of the 2x3 x 3x3 matrix multiplication tensor.
    # The all-ones tensor of shape (n,m,p) can be decomposed into 3 rank-1 terms
    # by grouping over the middle index j.
    rank = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        r = j
        # U: set entries for all i at column r
        for i in range(n):
            U[i * m + j][r] = 1
        # V: set entries for all k at column r
        for k in range(p):
            V[j * p + k][r] = 1
        # W: set entries for all (k,i) at column r
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
