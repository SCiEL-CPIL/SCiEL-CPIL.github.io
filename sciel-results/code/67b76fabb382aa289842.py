"""Construct a rank-44 exact tensor decomposition for 2x3 by 3x8 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][c] = 1
                    V[a * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 core
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

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        u = S_U[r][i * 2 + a]
                        v = S_V[r][a * 2 + k]
                        w = S_W[r][k * 2 + i]
                        s += u * v * w
                    if s != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()
    if not _ok():
        return _schoolbook()

    # Build rank-44 candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: Four 2x2x2 blocks for A[0:2, 0:2] * B[0:2, :]
    for block_idx in range(4):
        k0 = 2 * block_idx
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    uval = S_U[r][li * 2 + la]
                    if uval != 0:
                        U[li * m + la][col] = uval
            for la in range(2):
                for lk in range(2):
                    vval = S_V[r][la * 2 + lk]
                    if vval != 0:
                        V[la * p + (k0 + lk)][col] = vval
            for lk in range(2):
                for li in range(2):
                    wval = S_W[r][lk * 2 + li]
                    if wval != 0:
                        W[(k0 + lk) * n + li][col] = wval
            col += 1
            if not _ok():
                return _schoolbook()

    # Part 2: A[0:2, 2] * B[2, :8] (schoolbook, 16 terms)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _ok():
                return _schoolbook()

    # Validate
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        R = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    s = 0
                    ur = U[i * m + a]
                    vr = V[a * p + k]
                    wr = W[k * n + i]
                    for c in range(R):
                        uv = ur[c]
                        if uv == 0:
                            continue
                        vv = vr[c]
                        if vv == 0:
                            continue
                        s += uv * vv * wr[c]
                    if s != 1:
                        return False
        return True

    cand = {"U": U, "V": V, "W": W}
    if _validate(cand):
        return cand
    return _schoolbook()
# EVOLVE-BLOCK-END
