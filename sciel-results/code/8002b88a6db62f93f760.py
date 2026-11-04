"""
Find a rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication.
The output factor follows W[k*n+i, r].
"""

def construct():
    n, m, p = 2, 4, 4
    rank = 32

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
