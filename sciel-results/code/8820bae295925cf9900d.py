"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication in the index space defined by the seed
    # appears to be the all-ones tensor, as the seed decomposes it using
    # rank-1 terms e_i,j \otimes e_j,k \otimes e_k,i for all i,j,k.
    # The rank of the all-ones tensor is 1.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
