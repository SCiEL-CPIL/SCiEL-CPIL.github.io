"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 3 by 4 matrix by a 4 by 5 matrix.

The task requires returning factor matrices U, V, W such that the tensor
decomposition exactly matches the matrix multiplication tensor.
The score is the rank (lower is better).

The schoolbook algorithm provides a decomposition of rank n*m*p.
For n=3, m=4, p=5, rank = 60.
Unless a specific known decomposition with lower rank and integer/half-integer
entries is available, the schoolbook decomposition is the safest valid solution.
"""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the decomposition
    # Each column c corresponds to a specific (i, j, k) triple
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
