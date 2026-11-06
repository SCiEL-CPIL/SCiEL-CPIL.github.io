import time
import random
import numpy as np

N = 20
FULL_MASK = (1 << N) - 1
DEADLINE = 150.0
SAFETY = 0.85


def _check(A):
    """Return (ok, count). ok is True iff A is K3,3-free."""
    S = A @ A
    T = S @ S
    # T[i,j] = number of length-2 paths i->x->y->j. A K3,3 with parts
    # {i,j,k} and {x,y,z} gives T[i,j]=T[i,k]=T[j,k]=3.
    M = (T == 3)
    for i in range(N):
        r = M[i]
        for j in range(i + 1, N):
            if not r[j]:
                continue
            c = r[j + 1:]
            if c.any():
                return False, 0
    return True, int(A.sum() // 2)


def _can_add(A, u, v):
    """True iff adding edge (u,v) keeps the graph K3,3-free."""
    A[u, v] = 1
    A[v, u] = 1
    ok, _ = _check(A)
    A[u, v] = 0
    A[v, u] = 0
    return ok


def _greedy(A):
    """Add every edge that can be added without creating a K3,3."""
    for u in range(N):
        for v in range(u + 1, N):
            if A[u, v] == 0 and _can_add(A, u, v):
                A[u, v] = 1
                A[v, u] = 1
    return A


def _spectral_init(seed):
    """Spectral heuristic: threshold a random symmetric matrix at its mean."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, N)
    X = (X + X.T) / 2.0
    A = (X > np.mean(X)).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def _remove_k33(A):
    """Remove one edge of each K3,3 until the graph is K3,3-free."""
    while True:
        ok, _ = _check(A)
        if ok:
            return A
        S = A @ A
        T = S @ S
        M = (T == 3)
        found = False
        for i in range(N):
            for j in range(i + 1, N):
                if not M[i, j]:
                    continue
                for k in range(j + 1, N):
                    if not M[i, k] or not M[j, k]:
                        continue
                    common = np.nonzero((A[i] & A[j] & A[k]) != 0)[0]
                    for x in common:
                        A[i, x] = 0
                        A[x, i] = 0
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            # Fallback: drop a random edge
            edges = np.argwhere(A)
            if len(edges) == 0:
                return A
            u, v = edges[random.randrange(len(edges))]
            A[u, v] = 0
            A[v, u] = 0


def _anneal(A, time_limit, rng):
    """Simulated-annealing over single edge flips."""
    start = time.time()
    best = A.copy()
    best_count = _check(best)[1]
    cur = A.copy()
    cur_count = _check(cur)[1]

    T0 = 25.0
    Tend = 0.05
    iters = 0
    total = 120000
    while time.time() - start < time_limit:
        iters += 1
        # pick a random non-diagonal pair
        u = rng.randrange(N)
        v = rng.randrange(N)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        cur[u, v] ^= 1
        cur[v, u] ^= 1
        ok, cnt = _check(cur)
        if not ok:
            cur[u, v] ^= 1
            cur[v, u] ^= 1
            continue
        dE = cnt - cur_count
        if dE >= 0 or (rng.random() < np.exp(dE / max(T0 * (Tend / T0) ** (iters / total), 1e-6))):
            cur_count = cnt
            if cnt > best_count:
                best_count = cnt
                best = cur.copy()
        else:
            cur[u, v] ^= 1
            cur[v, u] ^= 1
    return best, best_count


def _kick(A, rng, k):
    """Perturb by flipping k random edges, then repair and fill."""
    B = A.copy()
    for _ in range(k):
        u = rng.randrange(N)
        v = rng.randrange(N)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        B[u, v] ^= 1
        B[v, u] ^= 1
    B = _remove_k33(B)
    B = _greedy(B)
    return B


def construct():
    start = time.time()
    deadline = start + DEADLINE * SAFETY
    rng = np.random.RandomState(12345)

    best = None
    best_count = 0

    def consider(A):
        nonlocal best, best_count
        A = _greedy(A)
        A = _remove_k33(A)
        A = _greedy(A)
        ok, cnt = _check(A)
        if ok and cnt > best_count:
            best_count = cnt
            best = A.copy()

    # Phase 1: diverse spectral starts, each with a short anneal
    for s in range(60):
        if time.time() > deadline - 10:
            break
        A = _spectral_init(s)
        A = _remove_k33(A)
        A = _greedy(A)
        consider(A)
        rem = deadline - time.time()
        if rem > 8:
            A2, c2 = _anneal(A, min(rem - 6, 6.0), rng)
            if c2 > best_count:
                best_count = c2
                best = A2.copy()

    # Phase 2: kick + anneal from the best solution
    while best is not None and time.time() < deadline - 20:
        rem = deadline - time.time()
        if rem < 20:
            break
        A = _kick(best, rng, rng.randint(3, 8))
        consider(A)
        rem = deadline - time.time()
        if rem > 10:
            A2, c2 = _anneal(A, min(rem - 8, 8.0), rng)
            if c2 > best_count:
                best_count = c2
                best = A2.copy()

    # Phase 3: final polish
    if best is not None and time.time() < deadline - 5:
        best = _greedy(best)
        best = _remove_k33(best)
        best = _greedy(best)
        ok, cnt = _check(best)
        if ok:
            best_count = cnt

    if best is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if best[i, j]:
                edges.append([i, j])
    return edges
