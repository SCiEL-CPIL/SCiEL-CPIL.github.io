"""Structural rewrite: fix K14 edges, search values via greedy + simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 14
M = 35
MAXV = 10000
DEADLINE = 80.0  # wall-clock safety margin under the 90s kill

# Precompute the complete-graph edge index list (exactly 35 edges on 14 vertices)
EDGE_I = np.zeros(M, dtype=np.int64)
EDGE_J = np.zeros(M, dtype=np.int64)
_k = 0
for _i in range(N):
    for _j in range(_i + 1, N):
        EDGE_I[_k] = _i
        EDGE_J[_k] = _j
        _k += 1
assert _k == M


def _score(vals):
    """Return (K, nsums, nprods) for an int array vals of length N."""
    s = vals[EDGE_I] + vals[EDGE_J]
    p = vals[EDGE_I] * vals[EDGE_J]
    nsums = int(np.unique(s).size)
    nprods = int(np.unique(p).size)
    return max(nsums, nprods), nsums, nprods


def _greedy(vals, rng, iters, deadline):
    """Greedily replace a random index with a random candidate to reduce K."""
    vals = vals.copy()
    curK, _, _ = _score(vals)
    for _ in range(iters):
        if time.time() > deadline:
            break
        idx = rng.randrange(N)
        cand = rng.randint(1, MAXV)
        if cand == vals[idx] or cand in vals:
            continue
        old = vals[idx]
        vals[idx] = cand
        newK, _, _ = _score(vals)
        if newK <= curK:
            curK = newK
        else:
            vals[idx] = old
    return vals, curK


def _anneal(vals, rng, deadline):
    """Simulated annealing over single-index value replacements."""
    vals = vals.copy()
    curK, _, _ = _score(vals)
    best_vals = vals.copy()
    best_K = curK

    t = 4.0
    tmin = 0.02
    cool = 0.9985

    while time.time() < deadline:
        idx = rng.randrange(N)
        cand = rng.randint(1, MAXV)
        if cand == vals[idx] or cand in vals:
            continue
        old = vals[idx]
        vals[idx] = cand
        newK, _, _ = _score(vals)
        delta = newK - curK
        if delta <= 0 or rng.random() < np.exp(-delta / t):
            curK = newK
            if newK < best_K:
                best_K = newK
                best_vals = vals.copy()
        else:
            vals[idx] = old
        t = max(tmin, t * cool)
    return best_vals, best_K


def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)

    # Strong sum-friendly base: consecutive integers 1..14
    base = np.arange(1, N + 1, dtype=np.int64)

    # Greedy construction to reduce K quickly
    vals, curK = _greedy(base, rng, 2500, deadline)

    # Simulated annealing to escape local optima
    vals, bestK = _anneal(vals, rng, deadline)

    # Ensure plain Python ints and valid distinct positive values
    vlist = [int(x) for x in vals]
    if len(set(vlist)) != N or any(x < 1 or x > MAXV for x in vlist):
        # Fallback to base (should not happen)
        vlist = list(range(1, N + 1))

    edges = [[int(a), int(b)] for a, b in zip(EDGE_I.tolist(), EDGE_J.tolist())]
    return {'values': vlist, 'edges': edges}
# EVOLVE-BLOCK-END
