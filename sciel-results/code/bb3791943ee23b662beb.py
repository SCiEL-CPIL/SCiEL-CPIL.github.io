"""Biclique partition via greedy large-biclique selection with refinement."""
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _greedy_partition(adj, n, remaining_edges, seed=0):
    """Greedy: repeatedly pick a large valid biclique among remaining edges."""
    rng = random.Random(seed)
    # remaining_edges: set of frozenset({u,v})
    remaining = set(remaining_edges)
    bicliques = []
    # adjacency restricted to remaining edges
    rem_adj = [set() for _ in range(n)]
    for e in remaining:
        u, v = tuple(e)
        rem_adj[u].add(v)
        rem_adj[v].add(u)

    active = [v for v in range(n) if rem_adj[v]]
    while remaining:
        # score candidate bicliques: for each vertex u, consider L={u}, R = subset of rem_adj[u]
        # We want large bicliques. Try: pick a vertex u with high degree, R = rem_adj[u], then shrink L to common neighbors.
        # Better: enumerate pairs (u,v) in remaining, build biclique by intersection.
        best_biclique = None
        best_size = 0
        # candidate: for each u in active, R0 = rem_adj[u]; then L = {u} ∪ {w in R0 : rem_adj[w] ⊇ R0}
        # To keep it fast, iterate over high-degree vertices.
        cand_us = sorted(active, key=lambda x: -len(rem_adj[x]))
        for u in cand_us[:40]:
            R0 = rem_adj[u]
            if not R0:
                continue
            # L = {u} plus any w in R0 whose rem_adj contains R0
            L = [u]
            for w in R0:
                if w == u:
                    continue
                if R0.issubset(rem_adj[w]):
                    L.append(w)
            # R = R0 (since u's neighbors). But need every w in L to connect to all R.
            # By construction w in L has rem_adj[w] ⊇ R0, and u connects to R0. Good.
            R = list(R0)
            # size = number of remaining edges covered = |L|*|R| (all present)
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        if best_biclique is None:
            # fallback: single edge
            e = next(iter(remaining))
            u, v = tuple(e)
            best_biclique = ([u], [v])
        L, R = best_biclique
        # remove covered edges
        covered = 0
        for u in L:
            for v in R:
                if u == v:
                    continue
                e = frozenset((u, v))
                if e in remaining:
                    remaining.discard(e)
                    rem_adj[u].discard(v)
                    rem_adj[v].discard(u)
                    covered += 1
        bicliques.append((list(L), list(R)))
        active = [v for v in active if rem_adj[v]]
    return bicliques

def _is_valid(bicliques, adj, n, edges):
    """Check: every cross pair is an edge; cross-edge sets partition all edges exactly once."""
    seen = set()
    for L, R in bicliques:
        if not L or not R:
            return False
        for u in L:
            for v in R:
                if u == v:
                    return False
                if v not in adj[u]:
                    return False
                e = frozenset((u, v))
                if e in seen:
                    return False
                seen.add(e)
    all_edges = set(frozenset((a, b)) for a, b in edges)
    return seen == all_edges

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    edge_set = set(frozenset((a, b)) for a, b in edges)

    # seed baseline: star decomposition (valid)
    def star_decomp():
        out = []
        for u in range(n):
            vs = [v for a, b in edges if (a == u and b != u)]
            if vs:
                out.append(([u], vs))
        return out
    best = star_decomp()
    best_count = len(best)

    # try greedy with several seeds/orderings
    for seed in range(0, 400):
        if time.time() > deadline:
            break
        bl = _greedy_partition(adj, n, edge_set, seed=seed)
        if len(bl) < best_count and _is_valid(bl, adj, n, edges):
            best = bl
            best_count = len(bl)
        if best_count <= 8:
            break

    # refine: try to merge/replace - local search by trying to drop a biclique and re-cover its edges
    # Simple local improvement: for each biclique, try to re-express its edges using existing structure is complex.
    # Instead, do a few restarts with randomized greedy that biases differently.
    for seed in range(1000, 1400):
        if time.time() > deadline:
            break
        bl = _greedy_partition(adj, n, edge_set, seed=seed)
        if len(bl) < best_count and _is_valid(bl, adj, n, edges):
            best = bl
            best_count = len(bl)
        if best_count <= 8:
            break

    # convert to required types: list of [left, right] with Python ints
    result = [[list(map(int, L)), list(map(int, R))] for L, R in best]
    return result
