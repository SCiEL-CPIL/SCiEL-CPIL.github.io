"""
Task: Matrix multiplication 2,4,5
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix.
Output factor follows the official trace convention W[k*n+i,r].
Interface: construct() -> {'U', 'V', 'W'}
"""

import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 2x4 by 4x5 matrix multiplication.
    Rank = 2 * 4 * 5 = 40.
    U has shape (n*m, rank) = (8, 40)
    V has shape (m*p, rank) = (20, 40)
    W has shape (p*n, rank) = (10, 40)
    Entries are 0 or 1.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed, but could use numpy for speed if needed.
    # Given the small size, lists are fine and avoid import overhead if not strictly needed, 
    # though numpy is available. The seed uses lists.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, c] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, c] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, c] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
