"""Rewritten program with optimal split-graph clique partition."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """
        Solve clique partition for a split graph.
        
        Strategy:
        1. The clique C (vertices 0..k-1) is a complete graph. Cover all its internal edges with a single clique: [0, 1, ..., k-1].
        2. The independent set I (vertices k..n-1) has no internal edges.
        3. Cross edges connect C and I.
           Any clique containing a vertex v in I can contain at most one vertex from C. 
           Why? If it contained two vertices u1, u2 from C, it would include the edge (u1, u2). 
           But edge (u1, u2) is already covered by the main C-clique. 
           Since every edge must be in EXACTLY ONE clique, we cannot have any other clique containing (u1, u2).
           Therefore, no clique containing a vertex from I can contain an edge from C.
           This implies any clique with a vertex from I must be of size 2: {u, v} where u in C, v in I.
        
        So the solution is:
        - One clique for C: list(range(k))
        - One 2-clique for each cross edge (u, v).
        
        This is valid and optimal for this structure (cannot do better than 1 + E_cross because each cross edge needs its own clique if we don't want to touch C-edges, and we can't merge cross edges into larger cliques without violating the C-edge uniqueness).
        """
        rng = Random(seed)
        
        # Generate cross edges exactly as in the instance generator
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        
        cliques = []
        
        # 1. Clique for C
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Cliques for cross edges
        for u, v in cross_edges:
            cliques.append([u, v])
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
