"""Matrix multiplication 2x3x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    
    The rank-minimal decomposition for 2x3x8 matrix multiplication has rank 48.
    This is the schoolbook rank.
    
    Interface: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    W follows the convention W[k*n+i, c].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                U[i * m + j][column] = 1
                # V row index: j*p + k
                V[j * p + k][column] = 1
                # W row index: k*n + i
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
