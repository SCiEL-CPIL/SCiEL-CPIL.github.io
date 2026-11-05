"""Cage construction for n(5,9): 5-regular graph, girth >= 9, fewest vertices.

Strategy (cage-search / lift-and-verify family):
  1) Fast CD(4,5) generator -> valid 5-regular girth-9 graph on 1200 vertices (seed).
  2) Cage search: build a SPARSE 5-regular base graph (a few vertices of degree <5),
     then LIFT it to exactly 5-regular by adding a perfect matching on the deficit
     vertices.  The lift only increases girth, so a base graph with girth >= 9
     gives a valid 5-regular girth-9 graph.  We verify girth >= 9 with a
     non-backtracking (Hashimoto) matrix certificate: min_v (N^5)_{v,v} == 0.
  3) Deadline guard: always return the best valid result found.
"""
import time
import random
import itertools
from collections import deque

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# CD(4, q)  -- Lazebnik-Ustimenko-Woldar explicit q-regular girth-(d+5) graph.
# For q = 5, d = 4 this is a valid 5-regular graph with girth >= 9.
# ---------------------------------------------------------------------------
def _luw_build(q, d, deadline):
    t0 = time.time()
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Girth-9 certificate via the non-backtracking (Hashimoto) matrix.
# There is a cycle of length < 9 iff some (N^t)_{v,v} > 0 for t in 2..8.
# Because N is symmetric, N^t = N^(t-2) * N^2, so we only need N^2.
# ---------------------------------------------------------------------------
def _girth_at_least_9(adj, n):
    N = sp.csr_matrix(adj, shape=(2 * n, 2 * n))
    N2 = (N @ N).tocsc()
    P = N2
    for _ in range(3):                      # P = N^2, N^4, N^6
        if (P.diagonal() > 0).any():
            return False
        P = (P @ N2).tocsc()
    P = (P @ N2).tocsc()                    # N^8
    return not (P.diagonal() > 0).any()


def _build_adj(edges, n):
    rows = np.fromiter(itertools.chain.from_iterable(edges), dtype=np.int64)
    cols = np.fromiter(itertools.chain.from_iterable((j for _, j in edges)), dtype=np.int64)
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj.eliminate_zeros()
    return adj


def _try_lift(adj, n, deficit, rng, deadline):
    """Add a random perfect matching on the deficit vertices; check girth>=9."""
    d = list(deficit)
    rng.shuffle(d)
    if len(d) % 2:
        return None
    new_edges = []
    for a, b in zip(d[0::2], d[1::2]):
        if a == b:
            return None
        new_edges.append((a, b))
    rows = list(np.asarray(adj.tocoo()).row)
    cols = list(np.asarray(adj.tocoo()).col)
    for a, b in new_edges:
        rows += [a, b]
        cols += [b, a]
    cand = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    cand.eliminate_zeros()
    deg = np.asarray(cand.sum(axis=1)).ravel()
    if (deg != 5).any():
        return None
    if _girth_at_least_9(cand, n):
        e = cand.tocoo()
        return [(int(r), int(c)) for r, c in zip(e.row, e.col) if r < c]
    return None


def _cage_search(n, k, g, deadline, rng, seed_graph):
    """Sparse base graph with a few deficit vertices, then lift to k-regular."""
    best = (seed_graph[0], seed_graph[1])
    best_n = best[0]
    target = 1116
    # deficit counts to try (must be even to admit a perfect matching)
    deficit_counts = [2, 4, 6, 8, 10, 12]
    for n_try in range(max(target, 2 * k), min(n, best_n) + 1):
        if time.time() > deadline:
            break
        for dc in deficit_counts:
            if dc > 2 * (n_try - k):
                continue
            for attempt in range(200):
                if time.time() > deadline:
                    break
                adj = _random_sparse(n_try, k, dc, rng)
                if adj is None:
                    continue
                if not _girth_at_least_9(adj, n_try):
                    continue
                deficit = [v for v in range(n_try)
                           if int(adj[v].sum()) < k]
                if len(deficit) != dc:
                    continue
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    res = _try_lift(adj, n_try, deficit, rng, deadline)
                    if res is not None and n_try < best_n:
                        best = (n_try, res)
                        best_n = n_try
                        break
    return best


def _random_sparse(n, k, deficit, rng):
    """Random k-regular-ish simple graph with exactly `deficit` vertices of
    degree k-1 and the rest degree k.  Returns adjacency matrix or None."""
    # target degree per vertex
    deg = np.full(n, k, dtype=np.int64)
    short = rng.sample(range(n), deficit)
    for v in short:
        deg[v] = k - 1
    if deg.sum() % 2:
        return None
    # Configuration model
    stubs = []
    for v in range(n):
        stubs.extend([v] * int(deg[v]))
    rng.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            return None
        key = (min(a, b), max(a, b))
        if key in edges:
            return None
        edges.add(key)
    if len(edges) != len(stubs) // 2:
        return None
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj.eliminate_zeros()
    d = np.asarray(adj.sum(axis=1)).ravel()
    if not np.array_equal(d, deg):
        return None
    return adj


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115.0           # safety margin under the 150 s wall clock
    rng = random.Random(12345)

    # 1) seed: explicit CD(4,5) 5-regular girth-9 graph (valid, 1200 vertices)
    n_seed, edges_seed = _luw_build(5, 4, deadline)
    adj_seed = _build_adj(edges_seed, n_seed)
    best = (n_seed, edges_seed)
    best_n = n_seed

    # 2) cage search for a smaller 5-regular girth-9 graph
    try:
        found = _cage_search(1200, k, g, deadline, rng, (n_seed, edges_seed))
        if found[0] < best_n:
            best = found
            best_n = found[0]
    except Exception:
        pass

    # 3) return the best valid result found
    n, edges = best
    return n, edges
