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
        Clique partition for split graphs using the structure:
        K = {0, ..., k-1} is a clique, I = {k, ..., n-1} is independent.
        
        Strategy:
        - One clique K = {0, ..., k-1} covers all internal-K edges.
        - For each u in K, one clique C_u = {u} ∪ N_I(u) covers all cross-edges incident to u.
          Since I is independent, C_u is a valid clique (all I-vertices in C_u are adjacent to u,
          and no two I-vertices are adjacent to each other, so no extra edges are introduced).
        - No edge is covered twice: internal-K edges only in K; cross-edge (u,v) only in C_u.
        
        Total cliques: 1 + k (if k >= 2, else just the cross-edges handled appropriately).
        """
        cliques = []
        
        # 1. The full clique K covers all internal edges
        if k >= 2:
            cliques.append(list(range(k)))
        elif k == 1:
            # K has only one vertex, no internal edges; handle cross-edges via C_0
            pass
        
        # 2. For each u in K, build C_u = {u} + neighbors in I
        # We need to find, for each u in [0, k), which vertices v >= k are adjacent to u.
        # Build adjacency for cross-edges.
        # From the edges list, collect cross neighbors per u.
        cross_neighbors = {u: [] for u in range(k)}
        for u, v in edges:
            if u < k and v >= k:
                cross_neighbors[u].append(v)
        
        for u in range(k):
            neigh = cross_neighbors[u]
            if neigh:
                # C_u = [u] + sorted(neigh)
                clique = [u] + sorted(neigh)
                if len(clique) >= 2:
                    cliques.append(clique)
            # If neigh is empty, C_u would be [u] which has size 1, skip (no cross-edges to cover)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
