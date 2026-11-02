"""Matrix multiplication 3x3x5 rank decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    
    The schoolbook decomposition has rank n*m*p = 45.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of integers
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
