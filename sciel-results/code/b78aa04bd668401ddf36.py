"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return sorted(edges + cross_edges)

    def clique_partition(edges, n, k):
        # Identify cross edges
        cross_edges = [(u, v) for u, v in edges if u < k and v >= k]
        
        # Baseline: One clique for K, one for each cross edge
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Optimization: Check for vertices in I connected to all K
        # If such a vertex exists, use K U {v} as a clique
        # This covers all K edges and all cross edges to v
        # It saves |K|-1 cliques (replaces 1 K-clique and |K| pair-cliques with 1 mixed-clique)
        
        # Map vertex in I to its neighbors in K
        neighbors_in_k = {}
        for u, v in cross_edges:
            if v not in neighbors_in_k:
                neighbors_in_k[v] = set()
            neighbors_in_k[v].add(u)
        
        # Find v with largest neighborhood
        best_v = None
        max_deg = 0
        for v, nbrs in neighbors_in_k.items():
            if len(nbrs) > max_deg:
                max_deg = len(nbrs)
                best_v = v
        
        # If best_v is connected to all K, use the large mixed clique
        if best_v is not None and max_deg == k:
            # Remove the K clique from baseline
            cliques = [] 
            # Add the mixed clique
            cliques.append(list(range(k)) + [best_v])
            
            # Add pair cliques for all OTHER cross edges
            # We need to skip edges incident to best_v
            for u, v in cross_edges:
                if v != best_v:
                    cliques.append([u, v])
            
            return cliques
        else:
            # Standard baseline
            for u, v in cross_edges:
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
