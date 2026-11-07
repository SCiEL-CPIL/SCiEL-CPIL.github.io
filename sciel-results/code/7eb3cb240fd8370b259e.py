def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
        
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Separate internal and cross edges
        internal_edges = []
        cross_edges = []
        for u, v in edges:
            if u < k and v < k:
                internal_edges.append((u, v))
            else:
                # Since it's a split graph, any edge not in C is in C-I
                # Note: instance generates edges sorted.
                # u < k, v >= k
                cross_edges.append((u, v))
                
        # Check if any i in I is connected to all C
        # Build adjacency for I
        deg = [0] * n
        for u, v in cross_edges:
            # u is in C, v is in I
            deg[v] += 1
            
        max_deg = 0
        best_i = -1
        for i in range(k, n):
            if deg[i] > max_deg:
                max_deg = deg[i]
                best_i = i
                
        cliques = []
        
        if max_deg == k:
            # Use C U {best_i}
            clique_big = list(range(k)) + [best_i]
            cliques.append(clique_big)
            
            # Add size-2 cliques for all other cross edges
            for u, v in cross_edges:
                if v != best_i:
                    cliques.append([u, v])
        else:
            # Use C
            if k >= 2:
                cliques.append(list(range(k)))
            
            # Add size-2 cliques for all cross edges
            for u, v in cross_edges:
                cliques.append([u, v])
                
        results.append(cliques)
        
    return results
