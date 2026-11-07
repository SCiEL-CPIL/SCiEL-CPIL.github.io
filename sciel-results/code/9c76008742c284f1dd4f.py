def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        cliques = []
        # One clique for the entire C
        cliques.append(list(range(k)))
        
        # 2-cliques for each cross edge
        for (u, v) in sorted(edges):
            if u < k and v >= k:  # cross edge
                cliques.append([u, v])
        
        results.append(cliques)
    
    return results
