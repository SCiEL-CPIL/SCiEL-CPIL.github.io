"""Matrix multiplication 4x4 by 4x9: search for a low-rank exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 3.0


def _time_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)


def _validate(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    for a in range(n * m):
        for b in range(m * p):
            ja, ka = divmod(a, m)
            jb, kb = divmod(b, p)
            if ja != jb:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if ic != ja:
                    continue
                s = 0
                for t in range(r):
                    s += U[a][t] * V[b][t] * W[c][t]
                if s != 0:
                    return False
    return True


def _single_split(n, m, p, splits, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for (i, j, k) in splits:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    return U, V, W


def _two_split(n, m, p, splits, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for (i, j1, j2, k) in splits:
        U[i * m + j1][col] = 1
        U[i * m + j2][col] = 1
        V[j1 * p + k][col] = 1
        V[j2 * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    return U, V, W


def _try_single(n, m, p):
    best = None
    for n2 in range(1, n + 1):
        for m2 in range(1, m + 1):
            for p2 in range(1, p + 1):
                if n2 > n or m2 > m or p2 > p:
                    continue
                r = (n // n2) * (m // m2) * (p // p2) * (n2 * m2 * p2)
                if r >= n * m * p:
                    continue
                if best is not None and r >= best[0]:
                    continue
                splits = []
                for a in range(n // n2):
                    for b in range(m // m2):
                        for c in range(p // p2):
                            for i in range(n2):
                                for j in range(m2):
                                    for k in range(p2):
                                        splits.append((a * n2 + i, b * m2 + j, c * p2 + k))
                if len(splits) != r:
                    continue
                U, V, W = _single_split(n, m, p, splits, r)
                if not _time_ok():
                    return best
                if _validate(U, V, W):
                    best = (r, U, V, W)
    return best


def _try_two(n, m, p):
    best = None
    for n2 in range(1, n + 1):
        for m2 in range(1, m + 1):
            for p2 in range(1, p + 1):
                if n2 > n or m2 > m or p2 > p:
                    continue
                for n3 in range(1, n + 1):
                    for m3 in range(1, m + 1):
                        for p3 in range(1, p + 1):
                            if n3 > n or m3 > m or p3 > p:
                                continue
                            if n2 + n3 <= n and m2 + m3 <= m and p2 + p3 <= p:
                                r = (n // n2) * (m // m2) * (p // p2) * (n2 * m2 * p2) + \
                                    (n // n3) * (m // m3) * (p // p3) * (n3 * m3 * p3)
                                if r >= n * m * p:
                                    continue
                                if best is not None and r >= best[0]:
                                    continue
                                splits = []
                                for a in range(n // n2):
                                    for b in range(m // m2):
                                        for c in range(p // p2):
                                            for i in range(n2):
                                                for j1 in range(m2):
                                                    for j2 in range(m2):
                                                        if j1 != j2:
                                                            continue
                                                        for k in range(p2):
                                                            splits.append((a * n2 + i, b * m2 + j1, b * m2 + j2, c * p2 + k))
                                U, V, W = _two_split(n, m, p, splits, r)
                                if not _time_ok():
                                    return best
                                if _validate(U, V, W):
                                    best = (r, U, V, W)
    return best


def construct():
    n, m, p = DIMENSIONS
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
    fallback = (rank, U, V, W)

    best = fallback
    res = _try_single(n, m, p)
    if res is not None and res[0] < best[0]:
        best = res
    if _time_ok():
        res = _try_two(n, m, p)
        if res is not None and res[0] < best[0]:
            best = res
    return {"U": best[1], "V": best[2], "W": best[3]}
