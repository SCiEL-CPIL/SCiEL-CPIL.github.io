import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4

    # Deadline guard
    t0 = time.time()
    deadline = t0 + 120.0

    # Strassen 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Verify 2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i]) != 1:
                    return _schoolbook(n, m, p)

    # Kronecker product for 4x4
    # U4_raw = kron(U2, U2): row index r_out * 4 + r_in
    # We need U4[i*4+j] = U4_raw[K] where K corresponds to (i1,j1) outer, (i2,j2) inner
    perm_U = np.zeros(16, dtype=int)
    for i in range(4):
        for j in range(4):
            S = i * 4 + j
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            r_out = 2 * i1 + j1
            r_in = 2 * i2 + j2
            perm_U[S] = r_out * 4 + r_in

    perm_V = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            S = j * 4 + k
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            r_out = 2 * j1 + k1
            r_in = 2 * j2 + k2
            perm_V[S] = r_out * 4 + r_in

    perm_W = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            S = k * 4 + i
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            r_out = 2 * k1 + i1
            r_in = 2 * k2 + i2
            perm_W[S] = r_out * 4 + r_in

    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)

    U4 = U4_raw[perm_U]
    V4 = V4_raw[perm_V]
    W4 = W4_raw[perm_W]

    # Verify 4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if np.dot(U4[i*4+j], V4[j*4+k] * W4[k*4+i]) != 1:
                    return _schoolbook(n, m, p)

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _schoolbook(n, m, p):
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
