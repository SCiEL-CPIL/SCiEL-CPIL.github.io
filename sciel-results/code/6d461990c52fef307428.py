import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use rank 23 decomposition for 3x3
    # Coefficients are from a known source
    # I will use a simple rank-9 decomposition if possible, but rank 9 is not valid for 3x3.
    # The rank of 3x3 is at least 19.
    # I will use the trivial rank 27 for 3x3.
    # Total rank 54.
    
    # To improve, I will use a rank-23 decomposition.
    # I will hardcode the coefficients.
    
    # Since I don't have the coefficients, I will return the trivial decomposition.
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
