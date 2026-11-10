import time
import math
import random

_DEADLINE = 75.0

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _build_adj(N=256):
    adj = [set() for _ in range(N)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if _is_square(a + b):
                ia, ib = a - 1, b - 1
                adj[ia].add(ib)
                if ia != ib:
                    adj[ib].add(ia)
    return adj

def _greedy_min_deg(adj, N, rng=None):
    rem = set(range(N))
    sol = set()
    while rem:
        if rng is not None and len(rem) > 40:
            sample = rng.sample(list(rem), 40)
            v = min(sample, key=lambda x: len(adj[x] & rem))
        else:
            v = min(rem, key=lambda x: len(adj[x] & rem))
        sol.add(v)
        rem -= adj[v]
        rem.discard(v)
    return sol

def _try_add(adj, N, sol):
    added = False
    for v in range(N):
        if v not in sol and not (adj[v] & sol):
            sol.add(v)
            added = True
    return added

def construct():
    start = time.time()
    N = 256
    adj = _build_adj(N)
    
    best = set()
    rng = random.Random(777)
    
    # Deterministic greedy first
    g = _greedy_min_deg(adj, N, None)
    while _try_add(adj, N, g):
        pass
    best = set(g)
    
    # Randomized restarts
    iter_count = 0
    while time.time() - start < _DEADLINE - 5:
        iter_count += 1
        g = _greedy_min_deg(adj, N, rng)
        while _try_add(adj, N, g):
            pass
        if len(g) > len(best):
            best = set(g)
        # Local improvement: try 1-removal + re-add
        improved = True
        while improved and time.time() - start < _DEADLINE - 5:
            improved = False
            for v in list(best):
                if time.time() - start > _DEADLINE - 5:
                    break
                cand = best - {v}
                # add all non-conflicting
                for w in range(N):
                    if w not in cand and not (adj[w] & cand):
                        cand.add(w)
                if len(cand) > len(best):
                    best = cand
                    improved = True
                    break
    
    result = sorted(v + 1 for v in best)
    if not result:
        result = [1]
    return result
