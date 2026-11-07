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
        # For each independent vertex v, find its neighbors in C
        neighbors = {v: set() for v in range(k, n)}
        for (u, v) in edges:
            if u < k and v < k:
                pass  # C-C edge, covered by the C clique
            elif u < k:
                neighbors[v].add(u)
            elif v < k:
                neighbors[u].add(v)
            # else: both in I, shouldn't happen
        
        cliques = []
        # One clique for the entire clique part C
        if k >= 2:
            cliques.append(list(range(k)))
        # One clique per independent vertex with neighbors in C
        for v in range(k, n):
            if len(neighbors[v]) >= 1:
                cliques.append(sorted(neighbors[v]) + [v])
        
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
        neighbors = {v: set() for v in range(k, n)}
        for (u, v) in edges:
            if u < k and v < k:
                pass
            elif u < k:
                neighbors[v].add(u)
            elif v < k:
                neighbors[u].add(v)
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        for v in range(k, n):
            if len(neighbors[v]) >= 1:
                cliques.append(sorted(neighbors[v]) + [v])
        results.append(cliques)
    
    return results
