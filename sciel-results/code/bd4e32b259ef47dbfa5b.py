"""Rewritten program: exact rank-90 decomposition for 3x5 x 5x6 matrix multiplication."""

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3*5*6 = 90 (schoolbook / trivial tensor rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
