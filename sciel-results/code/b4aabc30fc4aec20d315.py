"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # 1. Big clique for all edges within C
        # This covers all (u, u') for u, u' in [0, k)
        if k >= 2:
            cliques.append(list(range(k)))
            
        # 2. Cover cross edges
        # Since we used the big clique for internal C edges,
        # any other clique can contain at most one vertex from C.
        # Thus, each cross edge (u, v) must be covered by a clique {u, v} 
        # or a larger clique containing u, v and only vertices from I.
        # But vertices in I are independent, so no clique can contain two vertices from I.
        # So the only possible cliques covering (u, v) are {u, v}.
        # (Unless we don't use the big clique, but that's complex).
        
        # We iterate over all cross edges
        for v in range(k, n):
            for u in range(k):
                if (u, v) in edges:
                    cliques.append([u, v])
                    
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
# EVOLVE-BLOCK-END
