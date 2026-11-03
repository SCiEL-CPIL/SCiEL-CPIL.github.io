"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication C = AB is T_{i,j,k} = 1.
    # This tensor has rank 1: T = 1 (vector of 1s) (x) 1 (vector of 1s) (x) 1 (vector of 1s).
    # U has shape (n*m, 1), V has shape (m*p, 1), W has shape (p*n, 1).
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
