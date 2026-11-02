import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140

    n, m, p = 4, 4, 8

    # Rank-16 decomposition for 4x4x4 (schoolbook 2x2 blocks)
    U_4 = np.zeros((16, 16), dtype=float)
    V_4 = np.zeros((16, 16), dtype=float)
    W_4 = np.zeros((16, 16), dtype=float)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_4[i * 4 + j, col] = 1
                V_4[j * 4 + k, col] = 1
                W_4[k * 4 + i, col] = 1
                col += 1

    rank = 32
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    for c in range(16):
        U[:, c] = U_4[:, c]
        for row in range(16):
            if V_4[row, c] != 0:
                j = row // 4
                k = row % 4
                V[j * 8 + k, c] = V_4[row, c]
        for row in range(16):
            if W_4[row, c] != 0:
                k = row // 4
                i = row % 4
                W[k * 4 + i, c] = W_4[row, c]

        U[:, 16 + c] = U_4[:, c]
        for row in range(16):
            if V_4[row, c] != 0:
                j = row // 4
                k = row % 4
                V[j * 8 + (k + 4), 16 + c] = V_4[row, c]
        for row in range(16):
            if W_4[row, c] != 0:
                k = row // 4
                i = row % 4
                W[(k + 4) * 4 + i, 16 + c] = W_4[row, c]

    return {"U": U, "V": V, "W": W}
