"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor for (n x m)(m x p) -> (n x p)
    # has T[i,j,k] = 1 for all valid i,j,k.
    # The all-ones tensor of size n*m x m*p x p*n has rank 1:
    # T[i,j,k] = 1 * 1 * 1
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
