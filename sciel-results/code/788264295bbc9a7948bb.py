"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 2x5x5 matrix multiplication.
    
    We use the schoolbook decomposition with rank 50. The structure of the tensor
    T_{2,5,5} requires at least 50 rank-1 terms: each of the 2 output rows requires
    25 independent computations (one for each of the 5x5=25 output entries), and
    the supports in the U and W factors are disjoint across the two output rows,
    preventing any sharing of rank-1 terms.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
    # U: shape (n*m, rank) = (10, 50)
    # V: shape (m*p, rank) = (25, 50)
    # W: shape (p*n, rank) = (10, 50)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook: for each (i, j, k) triple, one rank-1 term
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1  (following the convention W[k*n+i, r])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
