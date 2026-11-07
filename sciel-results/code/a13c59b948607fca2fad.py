import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        rng = Random(seed)
        
        # Generate N_K for each v in I
        # Must match instance generation exactly
        N_K = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N_K[v].add(u)
                    
        cliques = []
        
        # 1. Add clique K (vertices 0 to k-1)
        # This covers all edges within K
        if k >= 2:
            cliques.append(list(range(k)))
            
        # 2. Add cliques for cross edges
        # Use singletons {v, u} to avoid covering any K edges
        for v in range(k, n):
            for u in N_K[v]:
                cliques.append([v, u])
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            break
        results.append(clique_partition(*spec))
    
    return results
