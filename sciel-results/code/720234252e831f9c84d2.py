"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        # K is 0..k-1, I is k..n-1
        edges = []
        # Edges within K
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between K and I
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
                    cross_edges.append((u, v))
        return edges, cross_edges

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges, cross_edges = instance(n, k, p, seed)
        
        cliques = []
        # Add the clique K
        cliques.append(list(range(k)))
        
        # Add individual cross edges as cliques
        for u, v in cross_edges:
            cliques.append([u, v])
            
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
