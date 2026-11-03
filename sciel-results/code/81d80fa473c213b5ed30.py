"""Matrix multiplication 3,3,8: rank-minimal exact tensor decomposition search."""
import random
import time

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
SCHOOLBOOK = N * M * P  # 72


def _is_halfint(x):
    r = float(x) * 2.0
    return abs(r - round(r)) < 1e-9


def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0])
        for a in range(N * M):
            for b in range(M * P):
                acc = [0.0] * (P * N)
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    uv = u * v
                    if uv == 0:
                        continue
                    wc = W[c]
                    for k in range(P * N):
                        acc[k] += uv * wc[k]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                for kk in range(P * N):
                    expected = 1.0 if (kk % N == i and kk // N == j2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False


def _random_structured(rank, iters, deadline, rng, start_rank=None):
    """Random search for a sparse exact decomposition of a given rank."""
    best = None
    start_rank = start_rank if start_rank is not None else rank
    for it in range(iters):
        if time.time() > deadline:
            break
        # random sparse support: each of the three matrices gets a random subset of columns
        cols = list(range(start_rank))
        U = [[0] * start_rank for _ in range(N * M)]
        V = [[0] * start_rank for _ in range(M * P)]
        W = [[0] * start_rank for _ in range(P * N)]
        # assign a random small set of nonzero columns per row
        for a in range(N * M):
            nzc = rng.randint(1, 3)
            for c in rng.sample(cols, nzc):
                U[a][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        for b in range(M * P):
            nzc = rng.randint(1, 3)
            for c in rng.sample(cols, nzc):
                V[b][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        for kk in range(P * N):
            nzc = rng.randint(1, 3)
            for c in rng.sample(cols, nzc):
                W[kk][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        if _verify(U, V, W):
            best = (start_rank, U, V, W)
            return best
    return best


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s wall-clock kill
    rng = random.Random(12345)

    # Baseline: schoolbook (always valid)
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    best = (rank, U, V, W)

    # Attempt to improve via random structured search at lower ranks.
    # Try a few candidate ranks below schoolbook; verify each candidate.
    for cand_rank in range(40, 71):  # search lower ranks first
        if time.time() > deadline:
            break
        res = _random_structured(cand_rank, iters=20000, deadline=deadline, rng=rng)
        if res is not None:
            best = res
            break  # found a valid lower rank; stop (we want minimal)

    # If nothing found, fall back to the best valid (schoolbook).
    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}
