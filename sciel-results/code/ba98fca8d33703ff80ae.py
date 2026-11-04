import time
import numpy as np
import itertools

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Try to find a lower rank decomposition
    # This is a placeholder for a search algorithm
    # For now, return the schoolbook decomposition
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
