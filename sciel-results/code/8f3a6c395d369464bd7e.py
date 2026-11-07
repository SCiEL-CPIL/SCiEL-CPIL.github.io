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
        Clique partition for split graphs.
        K = {0, ..., k-1} is a clique.
        I = {k, ..., n-1} is an independent set.
        
        Strategy:
        1. Use the entire clique K as one clique. This covers all edges within K.
        2. For each cross edge (u, v) with u in K and v in I, create a 2-clique {u, v}.
           This is optimal given that K is used as a single clique, because any clique
           containing two K-vertices and an I-vertex would duplicate K-K edges.
        
        Total cliques: 1 (for K) + number of cross edges.
        """
        cliques = []
        
        # 1. The clique K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Cross edges: create a 2-clique for each
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
