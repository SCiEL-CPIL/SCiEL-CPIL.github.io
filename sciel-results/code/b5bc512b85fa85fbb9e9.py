import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
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
                
    # Check time
    if time.time() > deadline - 5.0:
        # Return best found (which is the schoolbook one)
        pass
        
    return {"U": U, "V": V, "W": W}
