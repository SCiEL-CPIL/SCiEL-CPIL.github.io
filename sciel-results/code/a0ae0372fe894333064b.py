import time
import math
import random
from scipy.optimize import linprog

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _bron_kerbosch(adj, n):
    adj = [set(a) for a in adj]
    max_c = []
    P = [set(range(n)) for _ in range(n)]
    X = [set() for _ in range(n)]
    R = set()

    def bk():
        if not P[0]:
            if len(R) >= 2:
                max_c.append(frozenset(R))
            return
        cand = [v for v in range(n) if v in P[0]]
        for v in cand:
            newP = [P[i] & adj[v] for i in range(n)]
            newX = [X[i] | {v} for i in range(n)]
            R.add(v)
            bk()
            R.discard(v)
            for i in range(n):
                P[i].add(v)
                X[i] = newX[i]

    bk()
    return max_c


def _solve_instance(n, p, seed):
    edges = _instance(n, p, seed)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    cliques = _bron_kerbosch(adj, n)
    cliques = [c for c in cliques if len(c) >= 2]
    if not cliques:
        return []

    # incidence: vertex -> set of clique indices
    v2c = [set() for _ in range(n)]
    for ci, c in enumerate(cliques):
        for v in c:
            v2c[v].add(ci)
    m = len(cliques)

    # ---- greedy seed (degree tie-break, then random shuffles) ----
    def greedy(order):
        rem = [set(c) for c in cliques]
        cnt = [0] * m
        for v in order:
            cs = v2c[v]
            if not any(cnt[ci] == 0 for ci in cs):
                continue
            gain = sum(1 for ci in cs if cnt[ci] == 0)
            if gain > 0:
                for ci in cs:
                    if cnt[ci] == 0:
                        cnt[ci] = 1
        return sum(cnt) == m, [v for v in order if v2c[v] and any(cnt[ci] > 0 for ci in v2c[v])]

    best = None
    # natural order
    ok, sol = greedy(list(range(n)))
    if ok:
        best = set(sol)
    # random shuffles
    rng = random.Random(1234)
    for _ in range(40):
        order = list(range(n))
        rng.shuffle(order)
        ok, sol = greedy(order)
        if ok and (best is None or len(sol) < len(best)):
            best = set(sol)
    if best is None:
        # fallback: all touched vertices
        touched = set()
        for c in cliques:
            touched |= c
        best = touched

    # ---- local search shrink ----
    def is_valid(S):
        cnt = [0] * m
        for v in S:
            for ci in v2c[v]:
                cnt[ci] += 1
        return all(cnt[ci] > 0 for ci in range(m))

    def shrink(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for v in sorted(S, key=lambda x: (-len(v2c[x]), x)):
                S.discard(v)
                if is_valid(S):
                    improved = True
                else:
                    S.add(v)
        return S

    best = shrink(best)

    # ---- randomized removal polish (multiple orders) ----
    def polish(S, rng2, iters):
        S = set(S)
        for _ in range(iters):
            progressed = False
            for v in list(S):
                if rng2.random() < 0.5:
                    continue
                S.discard(v)
                if is_valid(S):
                    progressed = True
                else:
                    S.add(v)
            if not progressed:
                break
        return S

    rng3 = random.Random(999)
    for _ in range(25):
        cand = polish(best, rng3, 2)
        if len(cand) < len(best):
            best = shrink(cand)

    return sorted(best)


def construct():
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > _DEADLINE - 30:
            # not enough time for a fresh instance; use a safe fallback
            touched = set()
            for u, v in _instance(n, p, seed):
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
            continue
        results.append(_solve_instance(n, p, seed))
    return results
