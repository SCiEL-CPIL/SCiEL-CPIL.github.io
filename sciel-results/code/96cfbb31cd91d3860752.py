"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Build the rank-54 tensor decomposition for M_{2,3,9} using
    # a direct-sum of three independent 2x1x3 slices.
    # For each j in 0..2, we factor the 2x1x3 matrix-vector product
    # (rank 3) independently, giving a total rank of 9 per j-block,
    # for a grand total of 54.
    #
    # Tensor convention: T[i,j,k] with i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2*3*9 = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        # This j-block covers columns i in 0..n-1 and k in 0..p-1
        # For each (i, k) pair in this j-slice, one term of the rank-3 slice
        for i in range(n):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                U[row_u][col] = 1
                V[row_v][col] = 1
                W[row_w][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
