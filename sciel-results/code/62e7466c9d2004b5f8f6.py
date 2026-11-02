"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A[i,j] to column 'column'
                U[i * m + j][column] = 1
                # V maps input B[j,k] to column 'column'
                V[j * p + k][column] = 1
                # W maps output C[i,k] to column 'column'
                # Convention: W[k*n + i, r]
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
