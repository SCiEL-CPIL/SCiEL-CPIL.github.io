def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
        
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Build adjacency for I vertices in C
        neighbors = [set() for _ in range(n)]
        internal_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((min(u,v), max(u,v)))
            else:
                # u < k, v >= k
                neighbors[v].add(u)
                
        # G_avail: set of internal edges still uncovered
        G_avail = set(internal_edges)
        
        cliques = []
        
        # Sort I vertices by degree descending
        I_vertices = [i for i in range(k, n) if neighbors[i]]
        I_vertices.sort(key=lambda x: len(neighbors[x]), reverse=True)
        
        def find_large_clique(candidates, graph_edges):
            # candidates is a set of vertices in C
            # graph_edges is a set of edges in C
            if not candidates:
                return set()
                
            # Build adjacency list for the induced subgraph
            adj = {v: set() for v in candidates}
            for u, v in graph_edges:
                if u in candidates and v in candidates:
                    adj[u].add(v)
                    adj[v].add(u)
            
            S = set()
            remaining = set(candidates)
            
            while remaining:
                # Pick vertex with max degree in remaining
                # If tie, pick arbitrary
                v = max(remaining, key=lambda x: len(adj[x] & remaining))
                
                # Add v to S
                S.add(v)
                
                # Update remaining: keep only vertices connected to v
                # and that are in remaining
                new_remaining = set()
                for u in remaining:
                    if u == v or u in adj[v]:
                        new_remaining.add(u)
                remaining = new_remaining
                
            return S

        for i in I_vertices:
            uncov_neigh = set(neighbors[i])
            while uncov_neigh:
                S = find_large_clique(uncov_neigh, G_avail)
                if not S:
                    # Should not happen if uncov_neigh is not empty
                    break
                clique = list(S) + [i]
                cliques.append(clique)
                
                # Remove internal edges of S from G_avail
                for u, v in itertools.combinations(S, 2):
                    edge = (min(u,v), max(u,v))
                    if edge in G_avail:
                        G_avail.remove(edge)
                        
                uncov_neigh -= S
                
        # Cover remaining internal edges in G_avail with C-only cliques
        while G_avail:
            active_vertices = set()
            for u, v in G_avail:
                active_vertices.add(u)
                active_vertices.add(v)
            
            S = find_large_clique(active_vertices, G_avail)
            if not S:
                # Should not happen
                break
            clique = list(S)
            cliques.append(clique)
            
            for u, v in itertools.combinations(S, 2):
                edge = (min(u,v), max(u,v))
                if edge in G_avail:
                    G_avail.remove(edge)
                    
        results.append(cliques)
        
    return results
