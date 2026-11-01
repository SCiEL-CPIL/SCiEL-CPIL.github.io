"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    # The matrix multiplication tensor for (n,m,p) has T[i,j,k] = 1 for all valid (i,j,k).
    # This is the all-ones tensor, which is a rank-1 tensor: outer product of three all-ones vectors.
    # T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] with r=0,
    # where U, V, W are all-ones columns.
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
