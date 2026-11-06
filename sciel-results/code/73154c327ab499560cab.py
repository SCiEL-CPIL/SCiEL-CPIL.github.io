"""
Combined approach:
- Fast K4 / independent-6 detection (bitmask based) from the current program.
- A robust, high-quality 35-vertex base graph (Paley-like construction).
- A structural extension strategy to 36 vertices using a constraint-driven
  neighborhood search (triangles / independent-5-sets), plus randomized local
  search to repair violations.
- A refinement loop with a strict deadline guard that always returns the best
  valid graph found.
"""

import time
import random
from itertools import combinations

_DEADLINE = None


def _k4_found(adj, n):
    """Return a 4-tuple forming a K4, or None."""
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None


def _ind6_found(adj, n):
    """Return a 6-tuple forming an independent set, or None."""
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            f = (higher & -higher).bit_length() - 1
                            return (a, b, c, d, e, f)
    return None


def _is_valid(adj, n):
    return _k4_found(adj, n) is None and _ind6_found(adj, n) is None


def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges


def _build_35_vertex_graph():
    """
    Build a 35-vertex (4,6)-Ramsey candidate using a Paley-like construction
    over Z_7 x Z_5.
    """
    n = 35
    adj = [0] * n
    S7 = {1, 2, 4}          # quadratic residues mod 7
    S5 = {1, 2, 3, 4}       # nonzero residues mod 5

    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(i1 + 1, 7):
                for j2 in range(5):
                    v2 = i2 * 5 + j2
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj in S5:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
    return adj


def _find_all_triangles(adj, n):
    triangles = []
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if (ab >> c) & 1:
                    triangles.append((a, b, c))
    return triangles


def _find_all_ind5(adj, n):
    allmask = (1 << n) - 1
    ind5 = []
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if (nabcd >> e) & 1:
                            ind5.append((a, b, c, d, e))
    return ind5


def _try_extend_35_to_36(adj35, deadline):
    """
    Try to extend the 35-vertex graph to 36 vertices by adding vertex 35
    with a carefully chosen neighborhood.
    """
    n_old = 35
    n_new = 36

    # Precompute constraints
    triangles = _find_all_triangles(adj35, n_old)
    ind5sets = _find_all_ind5(adj35, n_old)

    # We'll perform randomized local search on the adjacency of vertex 35.
    # Each trial: pick a random initial neighborhood, then iteratively fix
    # violations by flipping edges incident to 35 (or within the old graph
    # as a last resort).

    seeds = [12345, 67890, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888,
             99999, 10001, 20002, 30003, 40004, 50005, 60006, 70007, 80008, 90009]

    for seed in seeds:
        if time.time() > deadline:
            return None

        random.seed(seed)

        for attempt in range(12):
            if time.time() > deadline:
                return None

            adj = adj35[:] + [0]

            # Random initial neighborhood for vertex 35
            density = 0.35 + 0.3 * random.random()
            for i in range(n_old):
                if random.random() < density:
                    adj[35] |= (1 << i)
                    adj[i] |= (1 << 35)

            # Local search
            for step in range(20000):
                if time.time() > deadline:
                    return None

                k4 = _k4_found(adj, n_new)
                if k4 is None:
                    ind = _ind6_found(adj, n_new)
                    if ind is None:
                        return adj
                    a, b, c, d, e, f = ind
                    if 35 in (a, b, c, d, e, f):
                        others = [x for x in (a, b, c, d, e, f) if x != 35]
                        v = random.choice(others)
                        adj[35] |= (1 << v)
                        adj[v] |= (1 << 35)
                    else:
                        pairs = [(a, b), (a, c), (a, d), (a, e), (a, f),
                                 (b, c), (b, d), (b, e), (b, f),
                                 (c, d), (c, e), (c, f),
                                 (d, e), (d, f), (e, f)]
                        u, v = random.choice(pairs)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    a, b, c, d = k4
                    if 35 in (a, b, c, d):
                        others = [x for x in (a, b, c, d) if x != 35]
                        v = random.choice(others)
                        adj[35] &= ~(1 << v)
                        adj[v] &= ~(1 << 35)
                    else:
                        pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                        u, v = random.choice(pairs)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

    return None


def _local_search(adj, n, deadline, max_steps=40000):
    """
    Generic local search: flip edges to eliminate K4 and independent-6 violations.
    """
    for step in range(max_steps):
        if time.time() > deadline:
            return adj

        k4 = _k4_found(adj, n)
        if k4 is None:
            ind = _ind6_found(adj, n)
            if ind is None:
                return adj
            a, b, c, d, e, f = ind
            pairs = [(a, b), (a, c), (a, d), (a, e), (a, f),
                     (b, c), (b, d), (b, e), (b, f),
                     (c, d), (c, e), (c, f),
                     (d, e), (d, f), (e, f)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            a, b, c, d = k4
            pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    return adj


def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 110.0  # safety margin

    best_adj = None
    best_n = 0

    # 1) Build the known 35-vertex graph
    adj35 = _build_35_vertex_graph()
    if _is_valid(adj35, 35):
        best_adj = adj35
        best_n = 35

        # 2) Try to extend to 36
        result36 = _try_extend_35_to_36(adj35, _DEADLINE)
        if result36 is not None:
            return {'n': 36, 'edges': _edges_from_adj(result36, 36)}

    # 3) If 35 wasn't valid, or if we just want to refine, do local search
    #    on the 35-vertex graph to ensure validity (in case the construction
    #    produced a near-miss).
    if best_n == 35:
        refined = _local_search(best_adj[:], 35, _DEADLINE, max_steps=10000)
        if _is_valid(refined, 35):
            best_adj = refined
            best_n = 35

    # 4) Fallback: try smaller n via random init + local search
    for n in range(min(best_n - 1, 34), 16, -1):
        if time.time() > _DEADLINE:
            break
        for restart in range(5):
            if time.time() > _DEADLINE:
                break
            random.seed(1000 + n * 100 + restart)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            adj = _local_search(adj, n, _DEADLINE, max_steps=8000)
            if _is_valid(adj, n):
                best_adj = adj
                best_n = n
                break

    # 5) Final fallback: Paley graph on 17 vertices
    if best_n < 17:
        squares = {x * x % 17 for x in range(1, 17)}
        adj17 = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj17[i] |= (1 << j)
                    adj17[j] |= (1 << i)
        if _is_valid(adj17, 17):
            best_adj = adj17
            best_n = 17

    if best_adj is None:
        # Extremely unlikely; return a trivial valid graph
        return {'n': 1, 'edges': []}

    return {'n': best_n, 'edges': _edges_from_adj(best_adj, best_n)}
