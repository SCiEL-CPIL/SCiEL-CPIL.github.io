def construct():
    import time
    start_time = time.time()
    DEADLINE = 120  # seconds

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        # Edges within clique part
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        cliques = []
        
        # 1. Add the clique covering all edges within K
        # K is {0, ..., k-1}
        if k >= 2:
            cliques.append(list(range(k)))
            
        # 2. Add cliques for each cross edge
        # Cross edges are (u, v) where u < k and v >= k
        # We use size-2 cliques {u, v} to avoid covering edges within K
        cross_edges = []
        for u, v in edges:
            if (u < k and v >= k) or (u >= k and v < k):
                # Normalize so u is from K and v is from I
                if u >= k:
                    u, v = v, u
                cross_edges.append((u, v))
        
        # Sort for determinism (not strictly required but good practice)
        cross_edges.sort()
        
        for u, v in cross_edges:
            cliques.append([u, v])
            
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() - start_time > DEADLINE - 5:
            break
        results.append(solve_one(*spec))
    
    # If we ran out of time, fill remaining with trivial solution
    while len(results) < 3:
        spec = specs[len(results)]
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        results.append([[list(e)] for e in sorted(edges)])
    
    return results
