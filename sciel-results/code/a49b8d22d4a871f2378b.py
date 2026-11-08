import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    def _schoolbook():
        R = nm * mp * pn
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][c] = 1
                    V[a * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    if not _ok():
        return _schoolbook()

    # Strassen 2x2x2 core matrices
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    rank = 44
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # Part 1: 4 Strassen blocks for A[0:2, 0:2] * B[0:2, :]
    for block_idx in range(4):
        if not _ok():
            return _schoolbook()
        k0 = block_idx * 2

        for r in range(7):
            c = col + r

            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        U[li * m + la][c] = val

            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][c] = val

            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][c] = val

        col += 7

    if not _ok():
        return _schoolbook()

    # Part 2: Schoolbook for A[0:2, 2] * B[2, :8]
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

            if not _ok():
                return _schoolbook()

    # Fast validation using numpy
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)

    valid = True
    for i in range(n):
        for a in range(m):
            for k in range(p):
                # sum over rank of U[i*m+a, :] * V[a*p+k, :] * W[k*n+i, :]
                prod = U_np[i * m + a] * V_np[a * p + k] * W_np[k * n + i]
                s = int(np.sum(prod))
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return _schoolbook()

    return {"U": U, "V": V, "W": W}
