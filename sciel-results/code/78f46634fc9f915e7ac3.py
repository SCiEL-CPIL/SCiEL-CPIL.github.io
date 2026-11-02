import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # Rank 4 per k slice (schoolbook for 2x2 * 2x1)
    # Total rank = 13 * 4 = 52
    # This matches the baseline. 
    # To improve, I need rank 3.
    # I will try rank 3.
    
    # For each k, we use 3 terms.
    # Term 1: (a00 + a10) * b0
    # Term 2: (a01 + a11) * b1
    # Term 3: (a00 - a10) * b0 + (a01 - a11) * b1 ... this is not rank 1.
    
    # I will stick to rank 4 to ensure correctness.
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
