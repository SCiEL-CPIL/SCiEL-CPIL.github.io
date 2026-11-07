import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Fallback: return empty lists or whatever is valid? 
            # The task says "Any broken rule makes the whole result invalid".
            # So we must ensure we return a valid result.
            # If we run out of time, we can't guarantee validity if we stop mid-way.
            # But these graphs are tiny, so we won't run out.
            pass
            
        edges = instance(n, k, p, seed)
        
        # Build edge set for O(1) lookup
        edge_set = set()
        for u, v in edges:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
                
        cliques = []
        
        # 1. The clique K itself
        if k >= 2:
            cliques.append(list(range(k)))
            
        # 2. For each u in K, the clique {u} U N_I(u)
        for u in range(k):
            neighbors_in_I = []
            for v in range(k, n):
                if (u, v) in edge_set:
                    neighbors_in_I.append(v)
            
            if neighbors_in_I:
                clique = [u] + neighbors_in_I
                cliques.append(clique)
                
        results.append(cliques)
        
    return results
