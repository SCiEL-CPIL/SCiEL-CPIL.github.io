"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-10 decomposition of the 2x4 by 4x5 matrix multiplication tensor.
    
    The 2x4 matrix A is viewed as two independent 1x4 row vectors (a0, a1).
    Each row a_i multiplies the 4x5 matrix B (viewed as 5 column vectors b_k)
    to produce 1x5 result rows c_i = a_i * B.
    
    For each pair (i, k) where i in {0,1} and k in {0,1,2,3,4}, we create one
    rank-1 term:
      U[i*4 + j]      = a_i[j]   (input row i, column j of A)
      V[j*5 + k]      = b_k[j]   (input row j, column k of B)  
      W[k*2 + i]      = c_i[k]   (output column k, row i of C)
    
    Summing over all 10 terms gives the full matrix product.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = n * p  # 2 * 5 = 10
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 10
    V = [[0] * rank for _ in range(m * p)]  # 20 x 10
    W = [[0] * rank for _ in range(p * n)]  # 10 x 10
    
    col_idx = 0
    for i in range(n):  # 2 rows of A
        for k in range(p):  # 5 columns of B
            for j in range(m):  # 4 shared columns
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
            col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
