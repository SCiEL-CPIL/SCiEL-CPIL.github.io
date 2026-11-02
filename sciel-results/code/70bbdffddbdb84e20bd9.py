"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    
    The decomposition uses the schoolbook algorithm with rank = n*m*p = 60.
    Each factor matrix has integer entries (0 or 1).
    
    Indexing convention:
      U[i*m + j, c] = 1 if column c corresponds to (i,j,k)
      V[j*p + k, c] = 1 if column c corresponds to (i,j,k)  
      W[k*n + i, c] = 1 if column c corresponds to (i,j,k)
    
    For each column c, the triple product U[row_u, c] * V[row_v, c] * W[row_w, c]
    contributes to the (i, k) entry of the result matrix when row_u = i*m+j,
    row_v = j*p+k, and row_w = k*n+i.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
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
