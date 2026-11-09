"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes
        (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition: rank = n * m * p = 36
    # U[i*m + j, c] = 1 when c = i*m + j (row index for A)
    # V[j*p + k, c] = 1 when c = j*p + k (row index for B)  
    # W[k*n + i, c] = 1 when c = k*n + i (col index for C)
    #
    # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i,j,k), there is exactly one column c where all three are 1.
    
    rank = n * m * p  # 36
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 36
    V = [[0] * rank for _ in range(m * p)]  # 12 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
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
