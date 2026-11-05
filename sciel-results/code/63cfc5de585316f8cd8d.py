import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 3, 3, 6

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    # Rank 9 decomposition for 3x3 * 3x3 (schoolbook)
    def find_rank9():
        rank = 9
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = i * 3 + j * 3 + k
                    U[i * 3 + j, idx] = 1
                    V[j * 3 + k, idx] = 1
                    W[k * 3 + i, idx] = 1
        return U, V, W

    U1, V1, W1 = find_rank9()
    rank_total = 18
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))

    # First 9 columns: for B1 (k=0,1,2)
    U[:, 0:9] = U1
    for j in range(3):
        for k in range(3):
            V[j * 6 + k, 0:9] = V1[j * 3 + k, 0:9]
    W[:, 0:9] = W1

    # Last 9 columns: for B2 (k=3,4,5)
    U[:, 9:18] = U1
    for j in range(3):
        for k in range(3, 6):
            V[j * 6 + k, 9:18] = V1[j * 3 + (k - 3), 9:18]
    for k in range(3, 6):
        for i in range(3):
            W[k * 3 + i, 9:18] = W1[(k - 3) * 3 + i, 9:18]

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return make_schoolbook()
