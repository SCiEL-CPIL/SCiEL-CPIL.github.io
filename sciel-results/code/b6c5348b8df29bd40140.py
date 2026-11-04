"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook (rank-75) exact tensor decomposition for
    multiplying a 3x5 matrix by a 5x5 matrix.
    
    The multiplication tensor T has entries T[i,j,k] = 1 for the standard
    matrix product (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    
    Each rank-1 term corresponds to a single (i,j,k) triple:
      U[i*m + j, c] = 1
      V[j*p + k, c] = 1  
      W[k*n + i, c] = 1
    
    This gives rank = n*m*p = 75, which is the schoolbook rank.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 5 * 5 = 75
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
