"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _deadline = time.time() + 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges: u in [0,k), v in [k,n)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        
        # Safe Baseline Strategy:
        # 1. One clique for the entire C part (covers all internal edges of C).
        # 2. One clique {u, v} for each cross edge (u, v).
        # This ensures:
        # - All internal edges of C are covered exactly once (by the C clique).
        # - All cross edges are covered exactly once (by the {u,v} cliques).
        # - No clique has vertices from I with size > 1, so no internal edges of C are covered by cross cliques.
        # - No cross cliques cover internal edges of C (size of C-part is 1).
        # - Validity: 
        #   - C clique is valid (C is a clique).
        #   - {u, v} is valid if (u, v) is an edge.
        #   - Disjointness: 
        #     - Edges in C clique: pairs in C.
        #     - Edges in {u, v}: (u, v).
        #     - Intersection is empty.
        
        result = []
        if k >= 2:
            result.append(list(range(k)))
        
        for u, v in cross_edges:
            result.append([u, v])
            
        return result
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _deadline - 5:
            break
        results.append(solve_one(*spec))
    
    return results
# EVOLVE-BLOCK-END
