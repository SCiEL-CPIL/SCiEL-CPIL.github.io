"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Schoolbook decomposition of rank n*m*p = 36
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists for exact integer arithmetic
    U = [[0] * rank for _ in range(n * m)]  # 9 x 36
    V = [[0] * rank for _ in range(m * p)]  # 12 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (row of A)
                # V index: j*p + k (col of B)
                # W index: k*n + i (row of C per trace convention)
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    result = {"U": U, "V": V, "W": W}
    
    # Check time before returning
    if time.time() > deadline:
        pass
    
    return result
# EVOLVE-BLOCK-END
