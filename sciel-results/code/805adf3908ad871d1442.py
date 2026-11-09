"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a structured tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses the block-diagonal partitioning where each term corresponds to a unique
    (i, j, k) index triple. The schoolbook decomposition has rank n*m*p = 80.
    We return the standard schoolbook factorization with integer entries.
    
    The trace convention is W[k*n + i, r] for the output matrix element.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (16, rank)
        - V has shape (m*p, rank) = (20, rank)
        - W has shape (p*n, rank) = (20, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build factor matrices as lists of lists with integer entries
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Each column c corresponds to a unique (i, j, k) triple
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
