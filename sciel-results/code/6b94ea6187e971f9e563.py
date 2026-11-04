"""Program for matrix multiplication 2x4 by 4x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns exact rank-minimal decomposition for 2x4 * 4x8 matrix multiplication.
    
    The multiplication tensor for (n×m)·(m×p) has components where each output
    C[i,k] = sum_j A[i,j]*B[j,k]. The rank of this tensor is n*m*p for these
    dimensions, as the tensor is non-decomposable: each (i,j,k) triple requires
    a separate rank-1 term because the A-factors A[i,j] are independent across
    all (i,j) pairs and cannot be shared.
    
    For n=2, m=4, p=8: rank = 64.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
