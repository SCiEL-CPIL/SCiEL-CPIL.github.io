"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(edges, n, k):
        """
        Efficient clique partition for split graphs.
        K = {0, ..., k-1} is a clique.
        I = {k, ..., n-1} is an independent set.
        
        Strategy:
        1. Use the entire clique K as one clique. This covers all edges within K.
        2. For each vertex v in I, find its neighbors in K.
           Since I is independent, any clique covering an edge (u, v) with u in K
           cannot contain any other vertex from K (otherwise it would duplicate an edge in K).
           Thus, each edge (u, v) must be covered by a clique of size 2: {u, v}.
        
        Total cliques: 1 (for K) + number of cross edges.
        """
        cliques = []
        
        # 1. The clique K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Cross edges
        # We need to identify which edges are cross edges (u < k <= v)
        # And create a clique [u, v] for each.
        
        # We can just iterate through the provided edges and filter.
        # Or reconstruct from adjacency if needed, but edges list is sufficient.
        for u, v in edges:
            if u < k and v >= k:
                cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
