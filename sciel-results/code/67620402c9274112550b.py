"""Search for triangle-free graphs on 12 vertices with min degree >= 3 that
minimize the number of distinct simple-cycle lengths.

Strategy:
  1. Enumerate 3-regular bipartite graphs (two K3,3 copies) and 3-regular
     5-circulant graphs (cycle-free by construction) on 12 vertices.
  2. For each candidate compute the set of distinct simple-cycle lengths.
  3. Return the candidate with the fewest distinct lengths (1 = all cycles
     the same length, e.g. a single 12-cycle).
"""
import time
import itertools
import random

_START = time.time()
_DEADLINE = 80.0  # safety margin before the 90 s kill

_N = 12
_MIN_DEG = 3
_GIRTH = 4


def _elapsed_ok():
    return (time.time() - _START) < _DEADLINE


def _cycle_lengths(adj):
    """Return the set of lengths of all simple cycles in the graph (adj is a
    list of sorted neighbour lists)."""
    lengths = set()
    n = len(adj)
    for s in range(n):
        stack = [(s, s, frozenset((s,)))]
        while stack:
            u, start, seen = stack.pop()
            for v in adj[u]:
                if v == start and len(seen) >= _GIRTH:
                    lengths.add(len(seen))
                elif v > start and v not in seen:
                    stack.append((v, start, seen | {v}))
    return lengths


def _distinct_count(adj):
    return len(_cycle_lengths(adj))


def _edges_to_adj(edges):
    adj = [[] for _ in range(_N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for a in adj:
        a.sort()
    return adj


def _bipartite_candidates():
    """Two K3,3 copies: partitions {0..5} and {6..11}; each left vertex has
    degree 3 within its copy (2 internal + 1 cross), so min degree = 3."""
    cands = []
    for L1 in itertools.combinations(range(6), 3):
        for L2 in itertools.combinations(range(6), 3):
            for p in itertools.permutations(range(6)):
                edges = []
                for i, l in enumerate(range(6)):
                    for j in range(3):
                        if j < 2:
                            edges.append((l, L1[j]))
                        else:
                            edges.append((l, p[i] + 6))
                for i, l in enumerate(range(6)):
                    for j in range(3):
                        if j < 2:
                            edges.append((l + 6, L2[j] + 6))
                        else:
                            edges.append((l + 6, p[i]))
                cands.append(edges)
    return cands


def _circulant_candidates():
    """3-regular 5-circulant graphs on 12 vertices. The generator {+/-1,+/-k}
    with k in {2,3,4,5} is triangle-free (no three generators sum to 0 mod 12)."""
    cands = []
    for k in (2, 3, 4, 5):
        edges = []
        for i in range(_N):
            for d in (1, k):
                edges.append((i, (i + d) % _N))
        cands.append(edges)
    return cands


def _random_candidates(rng, count):
    """Random 3-regular bipartite graphs (two K3,3 copies)."""
    cands = []
    for _ in range(count):
        try:
            L1 = rng.sample(range(6), 3)
            L2 = rng.sample(range(6), 3)
            p = rng.sample(range(6), 6)
            edges = []
            for i, l in enumerate(range(6)):
                for j in range(3):
                    if j < 2:
                        edges.append((l, L1[j]))
                    else:
                        edges.append((l, p[i] + 6))
            for i, l in enumerate(range(6)):
                for j in range(3):
                    if j < 2:
                        edges.append((l + 6, L2[j] + 6))
                    else:
                        edges.append((l + 6, p[i]))
            cands.append(edges)
        except Exception:
            continue
    return cands


def construct():
    best_edges = None
    best_score = None

    # 1. Deterministic structured candidates.
    for edges in _circulant_candidates():
        if not _elapsed_ok():
            break
        adj = _edges_to_adj(edges)
        if min(len(a) for a in adj) < _MIN_DEG:
            continue
        sc = _distinct_count(adj)
        if best_score is None or sc < best_score:
            best_score = sc
            best_edges = edges

    # 2. Exhaustive bipartite candidates (bounded, ~64800).
    for edges in _bipartite_candidates():
        if not _elapsed_ok():
            break
        adj = _edges_to_adj(edges)
        if min(len(a) for a in adj) < _MIN_DEG:
            continue
        sc = _distinct_count(adj)
        if best_score is None or sc < best_score:
            best_score = sc
            best_edges = edges
        if best_score == 1:
            break  # cannot do better than one distinct cycle length

    # 3. Random bipartite candidates to fill remaining time.
    rng = random.Random(12345)
    if best_score is None or best_score > 1:
        for _ in range(200000):
            if not _elapsed_ok():
                break
            edges = _random_candidates(rng, 1)[0]
            adj = _edges_to_adj(edges)
            if min(len(a) for a in adj) < _MIN_DEG:
                continue
            sc = _distinct_count(adj)
            if best_score is None or sc < best_score:
                best_score = sc
                best_edges = edges
            if best_score == 1:
                break

    if best_edges is None:
        # Fallback: the seed prism (always valid: triangle-free, min deg 3, cyclic).
        best_edges = []
        for i in range(6):
            for u, v in [(i, (i + 1) % 6), (i + 6, (i + 1) % 6 + 6), (i, i + 6)]:
                best_edges.append(tuple(sorted((u, v))))

    out = []
    seen = set()
    for u, v in best_edges:
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        out.append([int(u), int(v)])
    return out
