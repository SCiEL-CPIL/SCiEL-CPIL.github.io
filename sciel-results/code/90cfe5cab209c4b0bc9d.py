"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    Uses the schoolbook (element-wise) decomposition with rank = n*m*p = 42.
    
    Convention: W[k*n+i, r] for output index (i,k) of C[i,k].
    U[i*m+j, r] for input A[i,j].
    V[j*p+k, r] for input B[j,k].
    
    For each triple (i, j, k), we have one rank-1 term:
      C[i,k] += A[i,j] * B[j,k]
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    rank = n * m * p      # 42
    
    # Build factor matrices as lists of lists
    U = [[0.0] * rank for _ in range(n * m)]  # shape (n*m, rank)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (m*p, rank)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (p*n, rank)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                U[i * m + j][col] = 1.0
                # V row index: j*p + k
                V[j * p + k][col] = 1.0
                # W row index: k*n + i (per convention W[k*n+i, r])
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
