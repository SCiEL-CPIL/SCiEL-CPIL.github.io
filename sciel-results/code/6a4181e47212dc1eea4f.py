"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-112 decomposition using Row-Reuse Tensor Contraction Strategy.
    
    This approach partitions the 4x4x7 multiplication into four 1x4x7 
    row-vector multiplications. For each row i of A, we compute the 
    product of the row vector A[i,:] (length 4) with the matrix B (4x7).
    
    The 1x4x7 tensor is decomposed using the standard basis, as the 
    general matrix multiplication tensor R(1, m, p) is mp for m,p > 1 
    in the absence of specific structural constraints like Toeplitz.
    Thus, each row requires 4*7 = 28 multiplications.
    Total rank = 4 * 28 = 112.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Initialize matrices with rank = n * m * p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over each row of the output C (which corresponds to a row of A)
    for i in range(n):
        # For this row, we compute C[i, :] = A[i, :] * B
        # This involves 4 (from A) * 7 (from B columns) multiplications
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
