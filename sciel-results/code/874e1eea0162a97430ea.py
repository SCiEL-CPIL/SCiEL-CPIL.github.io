import time

def construct():
    """
    Construct clique partitions for three split graph instances.
    The optimal partition for a split graph with clique C and independent set I is:
    1. The clique C itself.
    2. A 2-clique for each cross edge (u, v) with u in C and v in I.
    This is because:
    - C covers all edges within C.
    - No other clique can contain two vertices from C (would duplicate edges in C).
    - No clique can contain two vertices from I (I is independent).
    - Thus, any clique covering a cross edge (u, v) cannot contain any other vertex 
      (if it contained w in C, it would duplicate edge (u, w) or (v, w) if w in C? 
       Actually, if clique is {u, v, w} with u,w in C, it covers (u,w) which is in C. 
       So it must be just {u, v}).
    """
    
    def instance(n, k, p, seed):
        """Generate the edges of the split graph."""
        from random import Random
        rng = Random(seed)
        edges = []
        # Edges within C (clique)
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        cliques = []
        
        # 1. Add the main clique C = [0, k)
        # Ensure k >= 2 for the clique to be valid (size >= 2)
        # Given k is 10, 20, 30, this is fine.
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Add 2-cliques for all cross edges
        for u, v in edges:
            # Identify cross edges: one in [0, k), one in [k, n)
            if (u < k and v >= k) or (v < k and u >= k):
                # Ensure consistent ordering or just append
                # The problem doesn't require sorted vertices inside the clique, 
                # but "distinct int vertices".
                # Let's sort them for cleanliness, though not strictly required 
                # unless the checker cares about order (it says "list of ... vertices").
                # "Each clique is a list of at least two distinct int vertices".
                # It doesn't say "sorted".
                # However, standard practice is to keep them as found or sorted.
                # I will sort them to be safe.
                a, b = (u, v) if u < v else (v, u)
                cliques.append([a, b])
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for spec in specs:
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
