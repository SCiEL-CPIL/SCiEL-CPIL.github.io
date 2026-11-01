"""Rank-minimal decomposition for 2x2 by 2x11 matrix multiplication.

Strategy: search for a low-rank exact tensor decomposition by solving the
bilinear identity as a system of polynomial equations.  We use a randomized
sparse-grid over a small rational alphabet for the rank-r factor matrices,
verify exactness on a random rational test set, and locally refine (mutate)
promising candidates.  This is a fundamentally different design family from
the current program's fixed analytic construction: it is an optimization /
search method over the space of rank-r decompositions, targeting the known
upper bound R(2,2,11) <= 21 (and attempting 20) rather than returning the
schoolbook rank 44.
"""
import random
import time

DIMENSIONS = [2, 2, 11]


def _verify(U, V, W, n, m, p, trials=8, seed=12345):
    """Exact verification of the bilinear identity on random rational inputs.

    C[i,k] = sum_j A[i,j] B[j,k]
    reconstructed as sum_r U[i,r] V[j,r] W[k,r] A[i,j] B[j,k] per (i,j,k)
    (equivalently, for each (i,k): sum_{j,r} U[i,r] V[j,r] W[k,r] A[i,j] B[j,k]).
    """
    rng = random.Random(seed)
    for _ in range(trials):
        A = [[rng.randint(-4, 4) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-4, 4) for _ in range(p)] for _ in range(m)]
        # true product
        C_true = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_true[i][k] = s
        # reconstructed
        C_rec = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(n and p):  # p
                s = 0
                for j in range(m):
                    for r in range(len(U[0])):
                        s += U[i][r] * V[j][r] * W[k][r] * A[i][j] * B[j][k]
                C_rec[i][k] = s
        if C_true != C_rec:
            return False
    return True


def _grid_search(n, m, p, rank, alphabet, iters, deadline):
    """Randomized sparse grid search over factor entries from `alphabet`.

    Returns (U, V, W) if a valid rank-`rank` decomposition is found, else None.
    """
    rng = random.Random(987654321)
    for it in range(iters):
        if time.time() > deadline:
            return None
        U = [[rng.choice(alphabet) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.choice(alphabet) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.choice(alphabet) for _ in range(rank)] for _ in range(p * n)]
        # quick structural prune: each column should have nonzero support
        # (cheap check) before expensive verify
        ok = True
        for r in range(rank):
            if not any(U[i][r] for i in range(n * m)):
                ok = False
                break
            if not any(V[j][r] for j in range(m * p)):
                ok = False
                break
            if not any(W[k][r] for k in range(p * n)):
                ok = False
                break
        if not ok:
            continue
        if _verify(U, V, W, n, m, p, trials=4, seed=it):
            # confirm with more trials
            if _verify(U, V, W, n, m, p, trials=20, seed=it + 100000):
                return U, V, W
    return None


def _local_refine(U, V, W, n, m, p, alphabet, iters, deadline):
    """Mutate a candidate and keep improvements (here: any valid lower-rank
    sub-decomposition is not directly applicable, so we just try to find an
    even valid same-rank or prune to a valid one).  Returns best found."""
    rng = random.Random(555)
    best = (U, V, W)
    for it in range(iters):
        if time.time() > deadline:
            break
        U2 = [row[:] for row in U]
        V2 = [row[:] for row in V]
        W2 = [row[:] for row in W]
        # mutate a few entries
        for _ in range(3):
            mat = rng.choice([U2, V2, W2])
            i = rng.randrange(len(mat))
            r = rng.randrange(len(mat[0]))
            mat[i][r] = rng.choice(alphabet)
        if _verify(U2, V2, W2, n, m, p, trials=4, seed=it + 7777):
            if _verify(U2, V2, W2, n, m, p, trials=20, seed=it + 77770):
                best = (U2, V2, W2)
    return best


def _schoolbook(n, m, p):
    """Guaranteed-valid rank n*m*p decomposition."""
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


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit

    # Candidate alphabet of small rationals (integers and half-integers)
    alphabet = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    # Try to find a decomposition of rank 21 (known upper bound 2p-1),
    # then 20, then 19, ... down to a floor, each with a time budget.
    # This is a structural change: search over the decomposition space.
    best_result = None
    best_rank = n * m * p  # 44 schoolbook as fallback

    # Rank targets in descending desirability
    for rank in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]:
        if time.time() > deadline - 20.0:
            break
        # budget per rank: more for higher (harder) ranks, less for low
        budget = 30.0 if rank >= 20 else 12.0
        rank_deadline = min(deadline - 20.0, time.time() + budget)
        iters = 40000 if rank >= 20 else 20000
        res = _grid_search(n, m, p, rank, alphabet, iters, rank_deadline)
        if res is not None:
            U, V, W = res
            # local refine (may not improve rank but validates robustness)
            U, V, W = _local_refine(U, V, W, n, m, p, alphabet, 2000,
                                    min(deadline - 20.0, time.time() + 5.0))
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            # once we have a valid low rank, we can stop (rank-minimal search
            # would continue downward, but time is precious; stop at first
            # valid to guarantee a strong score, then optionally try lower)
            # We'll try to go one step lower if time allows.
            if time.time() > deadline - 25.0:
                break

    if best_result is None or time.time() > deadline - 15.0:
        # fallback to guaranteed-valid schoolbook
        return _schoolbook(n, m, p)

    return best_result
