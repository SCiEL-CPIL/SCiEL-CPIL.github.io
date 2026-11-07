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

    def clique_partition(n, k, p, seed):
        """
        Find a clique partition for a split graph.
        Strategy: 
        1. Use the entire clique K as one clique to cover all edges within K.
        2. For every edge (u, v) between K and I, use a clique {u, v} to cover it.
        This ensures every edge is covered exactly once:
        - Edges in K are in the K-clique.
        - Edges in K-I are in their respective pair-cliques.
        - No pair-clique contains 2 vertices from K, so no K-edge is covered twice.
        - No pair-clique contains 2 vertices from I, so no I-edge (none exist) is covered.
        """
        cliques = []
        
        # 1. Add the main clique for K
        if k >= 2:
            cliques.append(list(range(k)))
        elif k == 1:
            # If k=1, K is a single vertex. It has no internal edges.
            # We don't need a clique for K itself, but we might need it for edges to I?
            # No, edges to I are handled by pair cliques.
            # A single vertex is not a valid clique (must have >= 2 vertices).
            pass
            
        # 2. Add pair cliques for edges between K and I
        rng = Random(seed)
        # The RNG sequence must match the instance generator exactly.
        # The generator iterates v from k to n-1, and for each v, u from 0 to k-1.
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    # Edge (u, v) exists
                    # Create clique {u, v}
                    cliques.append([u, v])
                    
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END
