def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        return cross

    result = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cross = instance(n, k, p, seed)
        
        # Map each independent vertex v to its neighbors in the clique part
        # This defines the maximal cliques involving v
        neighbors = {v: [] for v in range(k, n)}
        for u, v in cross:
            neighbors[v].append(u)
            
        cliques = []
        
        # The main clique on the first k vertices
        # This covers all edges within the first k vertices
        if k >= 2:
            cliques.append(list(range(k)))
            
        # For each vertex v in the independent set, if it has any edges,
        # the maximal clique N(v) U {v} covers all edges incident to v.
        # Since no two independent vertices are connected, cliques for different v
        # cannot be merged. Thus, using the maximal clique for each v with deg > 0
        # is optimal.
        for v in range(k, n):
            if neighbors[v]:
                # N(v) U {v}
                # Convert to list of ints
                clique = [int(x) for x in neighbors[v]]
                clique.append(int(v))
                cliques.append(clique)
                
        result.append(cliques)
        
    return result
