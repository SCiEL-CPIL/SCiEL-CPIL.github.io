def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency for I vertices
        # N(v) is the set of neighbors in K
        N = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        
        cliques = []
        
        # Strategy A:
        # 1. Cover all cross edges with 2-cliques {v, u}
        # 2. Cover all core edges with 1 clique K
        
        # Cross edges
        for v in range(k, n):
            for u in N[v]:
                cliques.append([u, v])
        
        # Core edges
        # The set of vertices in K is {0, ..., k-1}
        # They form a clique
        if k >= 2:
            cliques.append(list(range(k)))
        
        return cliques

    results = []
    specs = [(30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)]
    
    for spec in specs:
        results.append(solve_instance(*spec))
        
    return results
