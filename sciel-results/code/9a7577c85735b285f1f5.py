"""Find rank-minimal exact tensor decomposition for 3x4 x 4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS


def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition of rank n*m*p
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
