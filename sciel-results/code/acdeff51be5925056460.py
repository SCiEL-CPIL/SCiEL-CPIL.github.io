"""Rewritten program with optimized split-graph clique partition using biclique grouping."""
# EVOLVE-BLOCK-START
import random

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        rng = random.Random(seed)
        
        # Generate cross edges adjacency
        # C vertices: 0 to k-1
        # I vertices: k to n-1
        adj = [set() for _ in range(k)]
        
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
        
        # Find S_common: vertices in I connected to ALL vertices in C
        if k > 0:
            s_common = adj[0].copy()
            for u in range(1, k):
                s_common.intersection_update(adj[u])
        else:
            s_common = set()
        
        cliques = []
        
        # Main clique: C union S_common
        # C is list(range(k)), S_common is set of ints
        main_clique = list(range(k)) + sorted(list(s_common))
        if len(main_clique) >= 2:
            cliques.append(main_clique)
            
        # Star cliques for remaining edges
        for u in range(k):
            remaining = adj[u] - s_common
            if remaining:
                star_clique = [u] + sorted(list(remaining))
                # star_clique size is at least 2 because remaining is not empty
                cliques.append(star_clique)
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
