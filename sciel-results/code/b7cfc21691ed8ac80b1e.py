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
            cliques = []
            for (u, v) in edges:
                if u < k and v < k:
                    cliques.append([u, v])
                elif u < k and v >= k:
                    cliques.append([u, v])
                else:
                    cliques.append([v, u])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        # Build adjacency for I-vertices
        # adj_I[i] is the list of neighbors in C for vertex i in I
        adj_I = {i: [] for i in range(k, n)}
        for (u, v) in edges:
            if u < k and v >= k:
                adj_I[v].append(u)
            elif u >= k and v < k:
                adj_I[u].append(v)

        cliques = []
        
        # 1. Cover all edges within C with cliques of size 2
        for u in range(k):
            for v in range(u + 1, k):
                cliques.append([u, v])
                
        # 2. Cover cross edges
        # For each i in I, if it has neighbors in C, create a clique N(i) U {i}
        for i in range(k, n):
            neighbors = adj_I[i]
            if neighbors:
                clique = [i] + neighbors
                cliques.append(sorted(clique))
            # If neighbors is empty, i has no cross edges, so no clique needed for i

        results.append(cliques)

    return results
