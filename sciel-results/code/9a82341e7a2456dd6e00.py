"""Schoolbook decomposition with deadline guard for 2x2 by 2x14 matrix multiply."""
import time

DIMENSIONS = [2, 2, 14]
DEADLINE = 150.0
START_TIME = time.time()


def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 56
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


if __name__ == "__main__":
    import numpy as np
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    print("Max error:", max_err)
    assert max_err < 1e-9
    print("Rank:", U.shape[1])
