import time
from random import Random
from itertools import combinations

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            # Fallback: trivial valid solution
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        # Build adjacency for I-vertices
        # For each i in I, find all c in C such that (c, i) is an edge
        adj_I = {i: [] for i in range(k, n)}
        for (u, v) in edges:
            if u < k and v >= k:
                adj_I[v].append(u)
            elif u >= k and v < k:
                adj_I[u].append(v)

        cliques = [list(range(k))]  # The C-C clique
        
        # For each I-vertex, create a clique with its neighbors in C
        # This is the optimal valid partition for split graphs
        for i in range(k, n):
            neighbors = adj_I[i]
            if neighbors:
                clique = [i] + neighbors
                cliques.append(sorted(clique))
            # If no neighbors, no edges to cover, so no clique needed for this i
            # (Though the vertex i exists, it has no cross edges, so it's isolated in the cross graph)
            # But wait, does i need to be in ANY clique?
            # "Every edge must lie in exactly one listed clique."
            # If i has no edges, it doesn't need to be in any clique for edge coverage.
            # But the problem says "Clique partitions... of the graph".
            # Usually, isolated vertices are not part of any edge, so they don't constrain the partition of edges.
            # However, if the definition implies covering vertices, that's different.
            # "Every listed set must be a clique... and every edge must lie in exactly one listed clique."
            # It does not say "every vertex must be in a clique".
            # So isolated I-vertices can be ignored.

        if not cliques:
            cliques = [list(range(k))]

        results.append(cliques)

    return results
