"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-54 (schoolbook) decomposition for 2x3 by 3x9 matrix multiplication.
    
    The decomposition is:
      U: shape (n*m, rank) = (6, 54)
      V: shape (m*p, rank) = (27, 54)
      W: shape (p*n, rank) = (18, 54)
    
    For each (i, j, k) in range(n) x range(m) x range(p), we assign one column r:
      U[i*m + j, r] = 1
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1
    
    This ensures that for any A (n x m) and B (m x p):
      C[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j} sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
             = sum_{j} A[i,j] * B[j,k]
    since for each (i,j,k), exactly one r has all three entries = 1, and that r contributes A[i,j]*B[j,k].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
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
