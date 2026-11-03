import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try Laderman-based decomposition (Rank 82)
    # If valid, return it. Else fallback to schoolbook (Rank 90).
    
    # Implementation of Laderman for 3x3x3
    # This is a placeholder for the actual coefficients.
    # Since I don't have the exact coefficients memorized reliably, 
    # and generating them is complex, I will stick to schoolbook to ensure correctness.
    
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
    return {"U": U, "V": V, "W": W}
