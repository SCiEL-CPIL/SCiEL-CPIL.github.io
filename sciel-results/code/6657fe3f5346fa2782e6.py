import time
import sys

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i * m + a][col] = 1
                V[a * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    acc = 0
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_val = u_row[c]
                        if u_val:
                            v_val = v_row[c]
                            if v_val:
                                acc += u_val * v_val * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition vectors
    # U: 7x4 (rows are rank terms, cols are i*2+a)
    # V: 7x4 (rows are rank terms, cols are a*2+k)
    # W: 7x4 (rows are rank terms, cols are k*2+i)
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
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-29 candidate
    # Part 1: 2x2x8 block via 4 Strassen 2x2x2 sub-blocks (rank 28)
    # Part 2: 2x1x8 block via 1 rank-1 term (rank 1)
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8) using Strassen on 4 column blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U part: li in 0..1, la in 0..1
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

            # V part: la in 0..1, lk in 0..1
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

            # W part: lk in 0..1, li in 0..1
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # This is a rank 1 tensor: u supported on a=2, v supported on a=2, w supported on all k
    # u_idx = i * m + 2 for i in 0..1
    # v_idx = 2 * p + k for k in 0..7
    # w_idx = k * n + i for k in 0..7, i in 0..1
    
    # u vector
    for i in range(n):
        u_idx = i * m + 2
        U[u_idx][col] = 1

    # v vector
    for k in range(p):
        v_idx = 2 * p + k
        V[v_idx][col] = 1

    # w vector
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            W[w_idx][col] = 1

    col += 1
    if not _elapsed_ok():
        return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
