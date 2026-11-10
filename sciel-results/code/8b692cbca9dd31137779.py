"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Strategy:
        # 1. One clique containing all vertices in the clique part (0..k-1).
        #    This covers all edges within the clique part exactly once.
        # 2. For every edge between the clique part and independent part,
        #    create a 2-clique (edge). This covers cross-edges exactly once.
        #    Since independent set vertices have no edges among themselves,
        #    no other edges exist.
        
        # All edges within clique part: one big clique
        clique_part = list(range(k))
        cliques = [clique_part]
        
        # Cross edges: each becomes its own 2-clique
        for u, v in edges:
            if u < k and v >= k:
                cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return results
# EVOLVE-BLOCK-END
