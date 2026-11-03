"""Combined program: Fixed schoolbook decomposition for 3x3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook decomposition
    # This provides a valid baseline with integer entries.
    rank = n * m * p
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
                
    # Verify time
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
        
    # Attempt to improve using a known block structure if time permits
    # For now, return the safe schoolbook solution.
    # Future work: Implement Strassen-based blocks for 3x3x3 and 3x3x4.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
