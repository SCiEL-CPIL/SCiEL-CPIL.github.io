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
        Valid clique partition for split graphs.
        
        Strategy:
        1. One clique for the entire K (covers all edges in K).
        2. One clique {v} ∪ N(v) for each independent vertex v with at least one neighbor in K.
        
        This ensures:
        - Every edge in K is in exactly one clique (the K clique).
        - Every cross edge (u, v) is in exactly one clique (the clique for v).
        - All cliques are valid (K is a clique, {v} ∪ N(v) is a clique since N(v) ⊆ K and K is a clique).
        """
        cliques = []
        
        # 1. The clique K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Build adjacency for independent vertices
        # For each independent vertex v >= k, find all u < k such that (u, v) is an edge
        neighbors = {}
        for u, v in edges:
            if u < k and v >= k:
                if v not in neighbors:
                    neighbors[v] = set()
                neighbors[v].add(u)
        
        # 2. For each independent vertex with at least one neighbor, create a clique
        for v in sorted(neighbors.keys()):
            if v >= n:
                continue
            nbrs = neighbors[v]
            if len(nbrs) >= 1:
                # Clique is {v} ∪ nbrs, must have at least 2 vertices
                clique = list(nbrs) + [v]
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
