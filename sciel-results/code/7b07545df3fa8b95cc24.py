"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication M_{n,m,p} has entries T_{ijk} = 1 for all i,j,k.
    # This is because C_{ik} = sum_j A_{ij} B_{jk}, so the coefficient of each A_{ij} B_{jk} term is 1.
    # The rank of the all-ones tensor J_{n x m x p} is 1.
    # We provide a rank 1 decomposition.
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
