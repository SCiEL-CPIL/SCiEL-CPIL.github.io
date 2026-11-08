"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return factor matrices U, V, W for the 3x4x8 matrix multiplication tensor.
    
    U: shape (n*m, rank) = (12, rank)
    V: shape (m*p, rank) = (32, rank)  
    W: shape (p*n, rank) = (24, rank)
    
    The decomposition satisfies: for all i,j,k:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = 1 if this is a valid multiplication term, else 0.
    
    Schoolbook: rank = n*m*p = 96
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 96
    
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
