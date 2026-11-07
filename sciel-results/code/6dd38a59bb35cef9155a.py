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
        Find a valid clique partition for a split graph.
        Strategy:
        1. Use the full clique K (first k vertices) to cover all edges within K.
        2. For each cross edge (u, v) with u in K and v in I, create a clique [u, v].
        
        This ensures:
        - All edges within K are covered exactly once by the K clique.
        - All cross edges are covered exactly once by their respective [u, v] clique.
        - No edge is covered twice (K clique has no cross edges; [u, v] cliques have no K-K edges).
        - All cliques have at least 2 vertices (K has k>=2, [u, v] has 2).
        """
        # Generate the graph
        rng = Random(seed)
        
        cliques = []
        
        # Clique for K (only if k >= 2)
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each vertex in I, check neighbors in K
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    # Edge (u, v) exists. Add clique [u, v].
                    cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            break
        results.append(clique_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
