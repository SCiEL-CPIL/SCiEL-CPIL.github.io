"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_split(n, k, p, seed):
        rng = Random(seed)
        cliques = []
        
        # One clique for all C x C edges
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each v in I, if it has neighbors in C, add clique {v} + N(v)
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            if len(nbrs) >= 1:
                cliques.append([v] + nbrs)
        
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        result.append(solve_split(*spec))
    
    return result
# EVOLVE-BLOCK-END
