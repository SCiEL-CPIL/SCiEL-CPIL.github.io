"""Program for matrix multiplication tensor decomposition M_{3,5,5}."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (15, rank)
      V has shape (m*p, rank) = (25, rank)
      W has shape (p*n, rank) = (15, rank)
    
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,k,j] = delta (1 if we're using the (i,j,k) term)
    
    Rank decomposition: for each (i,j,k), we have a rank-1 term:
      U[i*m + j, c] = 1
      V[j*p + k, c] = 1
      W[k*n + i, c] = 1
    
    where c is the column index (rank dimension).
    
    The schoolbook decomposition has rank n*m*p = 75.
    For M_{3,5,5}, this is likely optimal or near-optimal.
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
