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
        Strategy: Singleton strategy.
        1. Use clique K to cover all edges within K.
        2. For each vertex v in I and each neighbor u in K, use clique {v, u}.
        This ensures edges (v, u) are covered exactly once.
        Edges within K are covered only by K, as {v, u} has size 2 and contains
        only one vertex from K.
        """
        rng = Random(seed)
        # Generate the graph to get neighborhoods
        # adj_K[i] stores neighbors in K for vertex v = k + i in I
        num_i = n - k
        adj_K = [set() for _ in range(num_i)]
        for i in range(num_i):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_K[i].add(u)
        
        # Build the partition
        cliques = []
        # Clique for K
        if k >= 2:
            cliques.append(list(range(k)))
        elif k == 1:
            # If k=1, no edges in K. But we might still need to handle I.
            # K is a clique of size 1. No edges.
            pass
        else:
            # k=0, no K.
            pass
            
        # Cliques for edges between K and I
        for i in range(num_i):
            v = k + i
            for u in adj_K[i]:
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

    # Convert to lists of lists of ints to satisfy type checking
    # The problem states: "sequences must be list or tuple... convert before returning"
    # My cliques are already lists of ints.
    
    return results
# EVOLVE-BLOCK-END
