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
        # 2. For each independent vertex v (k..n-1), form clique {v} ∪ N(v)
        #    where N(v) = {u in 0..k-1 : (u,v) is an edge}.
        #    Since the clique part is a complete graph, N(v) is a clique,
        #    so {v} ∪ N(v) is a valid clique. This covers ALL cross-edges
        #    incident to v in one clique.
        #    Each cross-edge (u,v) appears in exactly one such clique (the one
        #    for vertex v), and the big clique only contains clique-part vertices,
        #    so no edge is covered twice.
        
        cliques = []
        
        # 1. The main clique for the clique part
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Build adjacency: for each independent vertex, find its neighbors in clique part
        # Cross edges are (u, v) where u < k and v >= k
        neighbor_sets = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                neighbor_sets[v].add(u)
            elif v < k and u >= k:
                neighbor_sets[u].add(v)
        
        # 2. One clique per independent vertex that has at least one neighbor
        for v in range(k, n):
            nbrs = neighbor_sets[v]
            if len(nbrs) >= 1:
                # {v} ∪ N(v) must have at least 2 vertices for it to be a valid clique
                clique = [v] + sorted(nbrs)
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
