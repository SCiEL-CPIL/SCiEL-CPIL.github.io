"""Circulant (algebraic) search for a k-regular graph of girth >= g.

We build Cayley graphs on the cyclic group Z_n with a symmetric generating set
S (|S| = k, 0 not in S, S = -S) and verify the girth exactly via a
non-backtracking (directed-edge) BFS from a single vertex (vertex-transitivity
makes root 0 sufficient). We search n upward from the Moore bound and keep the
smallest n admitting a valid generator set, under a strict wall-clock deadline.
"""
import time
import random
from collections import deque

_DEADLINE = 120.0
_START = time.time()


def _remaining():
    return _DEADLINE - 4.0 - (time.time() - _START)  # 4s safety margin


def _girth_ok(adj, n, g):
    """Exact girth test for a vertex-transitive graph: non-backtracking BFS
    from vertex 0. Returns True iff no cycle of length < g exists."""
    # State: (prev, cur). Start by taking each edge out of 0.
    visited = set()
    dq = deque()
    for v in adj[0]:
        dq.append((0, v, 1))
        visited.add((0, v))
    while dq:
        prev, cur, depth = dq.popleft()
        if depth + 1 >= g:
            continue
        for nxt in adj[cur]:
            if nxt == prev:
                continue  # non-backtracking
            if nxt == 0:
                # closed a cycle of length depth + 1
                if depth + 1 < g:
                    return False
                continue
            st = (cur, nxt)
            if st not in visited:
                visited.add(st)
                dq.append((cur, nxt, depth + 1))
    return True


def _make_circulant(n, S):
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in S:
            adj[i].append((i + s) % n)
    return adj


def _random_gen_set(n, k, rng):
    """Symmetric generating set of size k for Z_n.
    k odd  -> n must be even; include self-inverse n/2 plus (k-1)/2 pairs.
    k even -> k/2 pairs.
    Bias toward small generators (long cycles -> larger girth).
    """
    if k % 2 == 1:
        if n % 2 != 0:
            return None
        S = [n // 2]
        need = k - 1
    else:
        S = []
        need = k
    half = need // 2
    # Pool of candidate "positive" generators, small ones first (better girth).
    pool = [x for x in range(1, n // 2 + 1) if x != n // 2]
    # Mix: take some small, some random, to diversify.
    if len(pool) <= half:
        chosen = pool[:]
    else:
        small = pool[:max(1, half // 2)]
        rest = pool[len(small):]
        chosen = small + rng.sample(rest, half - len(small))
    for x in chosen:
        S.append(x)
        S.append((-x) % n)
    S = list(dict.fromkeys(S))
    while len(S) > k:
        S.pop()
    while len(S) < k:
        x = rng.randrange(1, n)
        if x != 0 and x not in S:
            S.append(x)
            S.append((-x) % n)
    S = list(dict.fromkeys(S))[:k]
    if len(S) != k:
        return None
    return S


def construct(k, g):
    t0 = time.time()
    rng = random.Random(20240607)

    # Moore bound for odd girth g = 2r+1.
    r = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(r))

    if k % 2 == 1:
        n_start = moore if moore % 2 == 0 else moore + 1
        n_step = 2  # odd k needs even n for a loopless symmetric set
    else:
        n_start = max(moore, k + 1)
        n_step = 1

    n_cap = 4000
    best_n, best_edges = None, None
    n = n_start
    while n <= n_cap and _remaining() > 20:
        tried = set()
        # Tries scale with remaining budget.
        ntries = max(40, int(3000 * max(0.0, _remaining()) / _DEADLINE))
        for t in range(ntries):
            if _remaining() < 20:
                break
            S = _random_gen_set(n, k, rng)
            if S is None:
                continue
            key = frozenset(S)
            if key in tried:
                continue
            tried.add(key)
            adj = _make_circulant(n, S)
            if any(len(a) != k for a in adj):
                continue
            if _girth_ok(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n, best_edges = n, edges
                break
        if best_n is not None:
            break
        n += n_step

    if best_n is None:
        # Fallback: return a k-regular simple circulant (girth not guaranteed).
        n = n_start
        while True:
            S = _random_gen_set(n, k, rng)
            if S is not None:
                break
            n += n_step
        adj = _make_circulant(n, S)
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges

    return best_n, best_edges
