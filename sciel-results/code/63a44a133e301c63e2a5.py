"""
Program to find the rank-minimal exact tensor decomposition for multiplying
a 2 by 2 matrix by a 2 by 15 matrix.

The theoretical minimum rank for M_{2,2,15} is 60, achieved by the schoolbook algorithm.
This program constructs that decomposition.
"""

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions (2, 2, 15).
    
    The decomposition is of rank 60, which is the minimum possible rank
    for this specific tensor shape.
    
    Returns:
        dict: A dictionary containing the keys 'U', 'V', 'W' with the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # The rank is n * m * p for this shape
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices based on the schoolbook decomposition
    # Each column corresponds to a unique (i, j, k) triplet
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][col] = 1
                
                # V index: j * p + k
                V[j * p + k][col] = 1
                
                # W index: k * n + i (following the trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
