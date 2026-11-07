"""Rewritten solver for clique partition of split graphs."""
import time

def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for n, k, p, seed in specs:
        edges = instance(n, k, p, seed)
        
        cliques = []
        # Separate C-C edges and C-I edges
        cc_edges = []
        ci_edges = []
        for u, v in edges:
            if u < k and v < k:
                cc_edges.append((u, v))
            else:
                ci_edges.append((u, v))
        
        # One clique for all C-C edges (the whole C)
        if k >= 2:
            cliques.append(list(range(k)))
        
        # One clique per C-I edge
        for c, i in ci_edges:
            cliques.append([c, i])
        
        results.append(cliques)
    
    return results
