"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = []
        # Edges within the clique C (first k vertices)
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        
        # Edges between C and I (independent set)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        edges = instance(n, k, p, seed)
        
        # Strategy: Use the entire clique C as one clique.
        # Use 2-cliques for all cross edges.
        cliques = []
        
        # 1. The main clique C
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Cross edges as 2-cliques
        # We need to distinguish internal edges from cross edges.
        # Internal edges are (u, v) with u < v < k.
        # Cross edges are (u, v) with u < k <= v.
        
        for u, v in edges:
            if u < k and v < k:
                # Internal edge, already covered by C
                continue
            else:
                # Cross edge
                cliques.append([u, v])
        
        # Convert to lists of ints
        final_cliques = [list(map(int, c)) for c in cliques]
        results.append(final_cliques)
        
    return results
# EVOLVE-BLOCK-END
