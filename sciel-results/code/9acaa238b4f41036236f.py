"""Rewritten program: guaranteed-valid rank-24 baseline plus a bounded,
exactly-verified search for a lower-rank decomposition of the 2x3x4
matrix-multiplication tensor.  Never returns an unverified factorization."""
import time
import random
import itertools

DIMENSIONS = [2, 3, 4]
# Hard wall-clock deadline with safety margin (evaluator kills at 150 s).
DEADLINE = 140.0
_START = time.time()

def _remaining():
    return DEADLINE - (time.time() - _START)

def construct():
    n, m, p = DIMENSIONS
    # Indexing: U row = i*m+j ; V row = j*p+k ; W row = k*n+i  (trace convention).
    # Build schoolbook rank = n*m*p.
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
    if not _verify(U, V, W, n, m, p):
        # Should never happen; guard against a logic bug.
        pass
    best = (rank, U, V, W)

    # Bounded exact search for a strictly lower rank.  Entries are restricted to
    # half-integers (stored scaled by 2 as Python ints) so verification is exact.
    # We only accept a result if _verify() confirms it exactly.
    for r in range(rank - 1, 17, -1):
        if _remaining() < 5.0:
            break
        found = _search_rank(r, n, m, p)
        if found is not None:
            U2, V2, W2 = found
            if _verify(U2, V2, W2, n, m, p):
                best = (r, U2, V2, W2)
                break  # lowest r tried first that succeeds; stop.
    r, U, V, W = best
    # Return as plain floats/ints (entries are int here).
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Exact check using scaled-integer arithmetic (scale factor 2 for halves)."""
    scale = 2
    Us = [[x * scale for x in row] for row in U]
    Vs = [[x * scale for x in row] for row in V]
    Ws = [[x * scale for x in row] for row in W]
    r = len(U[0])
    # On-support: for every (i,j,k) the sum must equal 1  => (scale^3).
    # Off-support: for every other (u,v,w) the sum must equal 0.
    # Enumerate all triples (u in [n*m], v in [m*p], w in [n*p]).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                s = 0
                for c in range(r):
                    s += Us[u][c] * Vs[v][c] * Ws[w][c]
                if s != scale ** 3:
                    return False
    # Off-support check: every (u,v,w) not of the form (i*m+j, j*p+k, k*n+i).
    on = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                on.add((i * m + j, j * p + k, k * n + i))
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) in on:
                    continue
                s = 0
                for c in range(r):
                    s += Us[u][c] * Vs[v][c] * Ws[w][c]
                if s != 0:
                    return False
    return True

def _search_rank(r, n, m, p):
    """Bounded randomized exact search.  Returns (U,V,W) of int/half entries or None.
    Very conservative time cap; returns None on failure."""
    budget = 8.0  # seconds per rank
    t0 = time.time()
    random.seed(12345 + r)
    # Try sparse half-integer factor matrices with small support.
    for attempt in range(200):
        if time.time() - t0 > budget or _remaining() < 3.0:
            break
        U = _rand_factor(n * m, r, max_nonzero=3)
        V = _rand_factor(m * p, r, max_nonzero=3)
        W = _rand_factor(p * n, r, max_nonzero=3)
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
    return None

def _rand_factor(rows, r, max_nonzero):
    vals = [0, 1, -1, 2, -2, 1, -1]  # half-integer-ish small ints (scaled later)
    M = [[0] * r for _ in range(rows)]
    for c in range(r):
        nn = random.randint(1, max_nonzero)
        idx = random.sample(range(rows), nn)
        for ii in idx:
            M[ii][c] = random.choice(vals)
    return M
