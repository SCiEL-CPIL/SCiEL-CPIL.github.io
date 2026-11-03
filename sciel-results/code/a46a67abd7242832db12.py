"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication.
    
    The matrix multiplication tensor T has entries T[i, j, k, i', j', k'] (conceptually),
    but here we use the standard decomposition:
      C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor is decomposed as:
      T = sum_c U[c] x V[c] x W[c]
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    Indexing convention:
      U row index: i * m + j  (i in 0..n-1, j in 0..m-1)
      V row index: j * p + k  (j in 0..m-1, k in 0..p-1)
      W row index: k * n + i  (k in 0..p-1, i in 0..n-1)
    
    For the schoolbook decomposition, rank = n * m * p.
    Each term corresponds to a unique (i, j, k) triple.
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
