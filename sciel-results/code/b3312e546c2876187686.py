import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Using lists of lists for exact integer representation as required
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    # If we somehow run out of time (unlikely for 105 iterations), return partial?
                    # No, the loop is tiny. This guard is for hypothetical search.
                    # For schoolbook, it's instant.
                    pass
                    
    return {"U": U, "V": V, "W": W}
