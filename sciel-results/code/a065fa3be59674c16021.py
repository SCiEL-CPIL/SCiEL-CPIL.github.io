"""Clique partition of dense random graphs — greedy maximal-clique extraction."""
# EVOLVE-BLOCK-START
import time

def _elapsed(t0):
    return time.time() - t0

def _random_graph(n, p, seed):
    """Reproduce the verifier's random_graph(n, p, seed) edge list."""
    from random import Random
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

def _solve(n, p, seed, deadline):
    """
    Produce a clique partition of the random graph (n, p, seed).
    Strategy: greedy maximal-clique extraction.
      - Maintain a set of remaining edges.
      - Repeatedly pick an edge, expand it to a maximal clique in the
        remaining-edge graph, remove those edges, record the clique.
    """
    edges = _random_graph(n, p, seed)
    if not edges:
        return []

    # Build adjacency as sets of neighbors for fast intersection
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    remaining = set(edges)
    cliques = []

    while remaining:
        if _elapsed(t0) > deadline:
            # Fallback: each remaining edge as a K2
            for u, v in remaining:
                cliques.append([u, v])
            return cliques

        # Pick an edge: prefer edges where both endpoints have high remaining degree
        # For speed, just pick an arbitrary edge (first in the set)
        # but a better heuristic: pick edge with max (deg[u]+deg[v]) among remaining
        # For n<=64 this is fine.
        best_edge = None
        best_score = -1
        # To avoid O(E) scan every iteration being too slow, sample if large
        if len(remaining) > 200:
            # sample up to 100 edges
            import random as _rnd
            sample = list(_rnd.sample(list(remaining), 100))
        else:
            sample = list(remaining)
        for u, v in sample:
            sc = len(adj[u] & adj[v])  # number of common neighbors in original graph
            # Actually we want remaining-degree heuristic
            # Let's use: count how many remaining edges touch u or v
            # Simpler: just pick first
            break
        # Just pick the first edge for speed
        edge = next(iter(remaining))
        u0, v0 = edge

        # Build maximal clique containing u0, v0 in the graph induced by remaining edges
        # A vertex w can be added if (w,x) in remaining for all x in current clique
        clique = {u0, v0}
        # Candidate vertices: common neighbors of all vertices in clique w.r.t. remaining edges
        candidates = set()
        for x in clique:
            nbrs = set()
            for e in remaining:
                if x in e:
                    other = e[1] if e[0] == x else e[0]
                    nbrs.add(other)
            candidates &= nbrs

        # Greedily add vertices
        while candidates:
            # Pick vertex with most connections among candidates (greedy)
            best_v = max(candidates, key=lambda w: sum(1 for c in candidates if c != w and (w, c) in remaining or (c, w) in remaining))
            clique.add(best_v)
            # Update candidates: must be adjacent (in remaining) to best_v
            new_candidates = set()
            for c in candidates:
                if c == best_v:
                    continue
                if (best_v, c) in remaining or (c, best_v) in remaining:
                    new_candidates.add(c)
            candidates = new_candidates

        # Remove all edges within the clique from remaining
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                a, b = list(clique)[i], list(clique)[j]
                e = (a, b) if a < b else (b, a)
                remaining.discard(e)

        cliques.append(sorted(clique))

    return cliques

def construct():
    t0 = time.time()
    deadline = 130.0  # 150s limit, 20s margin

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _elapsed(t0) > deadline - 5:
            # Not enough time; use trivial partition
            edges = _random_graph(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        res = _solve(n, p, seed, deadline)
        results.append(res)

    return results
# EVOLVE-BLOCK-END
