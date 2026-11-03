"""Search for a rank-minimal exact tensor decomposition of 3x4 * 4x4 matmul."""
import time
import random
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N
FULL_RANK = NM * MP * PN


def _make_schoolbook():
    U = [[0] * FULL_RANK for _ in range(NM)]
    V = [[0] * FULL_RANK for _ in range(MP)]
    W = [[0] * FULL_RANK for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W


def _tensor_value(U, V, W, i, j, k):
    n, m, p = N, M, P
    rank = len(U[0])
    s = 0
    for c in range(rank):
        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
    return s


def _verify(U, V, W):
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i == 0 and j == 0 and k == 0) else 0
                if _tensor_value(U, V, W, i, j, k) != expected:
                    return False
    return True


def _verify_fast(U, V, W):
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i == 0 and j == 0 and k == 0) else 0
                s = 0
                for c in range(rank):
                    a = U[i * m + j][c]
                    b = V[j * p + k][c]
                    d = W[k * n + i][c]
                    if a and b and d:
                        s += a * b * d
                if s != expected:
                    return False
    return True


def _try_merge(U, V, W, c1, c2, alpha):
    """Merge column c2 into c1 with coefficient alpha."""
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u1 = U[i * m + j][c1]
                u2 = U[i * m + j][c2]
                v1 = V[j * p + k][c1]
                v2 = V[j * p + k][c2]
                w1 = W[k * n + i][c1]
                w2 = W[k * n + i][c2]
                d1 = u2 * v1 * w1 + u1 * v2 * w1 + u1 * v1 * w2
                d2 = u2 * v2 * w1 + u2 * v1 * w2 + u1 * v2 * w2
                d3 = u2 * v2 * w2
                delta = alpha * d1 + alpha * alpha * d2 + alpha * alpha * alpha * d3
                if delta != 0:
                    return None
    new_rank = rank - 1
    new_U = []
    for r in range(NM):
        row = []
        for c in range(rank):
            if c == c2:
                continue
            if c == c1:
                row.append(U[r][c1] + alpha * U[r][c2])
            else:
                row.append(U[r][c])
        new_U.append(row)
    new_V = []
    for r in range(MP):
        row = []
        for c in range(rank):
            if c == c2:
                continue
            if c == c1:
                row.append(V[r][c1] + alpha * V[r][c2])
            else:
                row.append(V[r][c])
        new_V.append(row)
    new_W = []
    for r in range(PN):
        row = []
        for c in range(rank):
            if c == c2:
                continue
            if c == c1:
                row.append(W[r][c1] + alpha * W[r][c2])
            else:
                row.append(W[r][c])
        new_W.append(row)
    return new_U, new_V, new_W


def _random_search(U, V, W, max_time, rng):
    """Try random merges to reduce rank."""
    best_rank = len(U[0])
    best_result = (U, V, W)
    deadline = time.time() + max_time

    while time.time() < deadline - 3:
        rank = len(U[0])
        if rank <= best_rank:
            found = False
            for c1 in range(rank):
                for c2 in range(c1 + 1, rank):
                    for alpha in [1, -1, Fraction(1, 2), Fraction(-1, 2),
                                  Fraction(1, 3), Fraction(-1, 3),
                                  2, -2, Fraction(2, 3), Fraction(-2, 3)]:
                        result = _try_merge(U, V, W, c1, c2, alpha)
                        if result is not None:
                            new_U, new_V, new_W = result
                            if _verify_fast(new_U, new_V, new_W):
                                U, V, W = new_U, new_V, new_W
                                if rank - 1 < best_rank:
                                    best_rank = rank - 1
                                    best_result = (new_U, new_V, new_W)
                                found = True
                                break
                if found:
                    break
            if not found and rank > 16:
                # Try random restarts
                for _ in range(50):
                    c1 = rng.randrange(rank)
                    c2 = rng.randrange(rank)
                    if c1 == c2:
                        continue
                    alpha = rng.choice([1, -1, 2, -2, Fraction(1, 2), Fraction(-1, 2),
                                        Fraction(1, 3), Fraction(-1, 3), Fraction(2, 3), Fraction(-2, 3)])
                    result = _try_merge(U, V, W, c1, c2, alpha)
                    if result is not None:
                        new_U, new_V, new_W = result
                        if _verify_fast(new_U, new_V, new_W):
                            U, V, W = new_U, new_V, new_W
                            if rank - 1 < best_rank:
                                best_rank = rank - 1
                                best_result = (new_U, new_V, new_W)
                            break
        else:
            U, V, W = best_result

    return best_result, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 140

    rng = random.Random(42)

    U, V, W = _make_schoolbook()
    best_result = (U, V, W)
    best_rank = FULL_RANK

    remaining = deadline - time.time() - 5
    if remaining > 10:
        U, V, W = _make_schoolbook()
        result, rank = _random_search(U, V, W, remaining, rng)
        if rank < best_rank:
            best_rank = rank
            best_result = result

    U, V, W = best_result
    if not _verify(U, V, W):
        U, V, W = _make_schoolbook()

    return {"U": U, "V": V, "W": W}
