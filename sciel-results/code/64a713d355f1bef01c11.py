def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def max_clique(vertices, edges):
        """
        Find a maximum clique in the subgraph induced by 'vertices' using 'edges'.
        Uses branch and bound.
        """
        if not vertices:
            return []
        
        # Sort vertices by degree in induced subgraph descending
        # This helps in finding large cliques early for better pruning
        deg = {v: 0 for v in vertices}
        for u in vertices:
            for w in vertices:
                if u < w and (u, w) in edges:
                    deg[u] += 1
                    deg[w] += 1
        
        sorted_verts = sorted(vertices, key=lambda x: -deg[x])
        
        best_clique = []
        
        def expand(current, candidates):
            nonlocal best_clique
            # Pruning
            if len(current) + len(candidates) <= len(best_clique):
                return
            
            if not candidates:
                if len(current) > len(best_clique):
                    best_clique = current[:]
                return
            
            # Heuristic: try candidates in order of degree?
            # Or just iterate.
            for i, v in enumerate(candidates):
                # Check if v connects to all in current
                if all((min(v, u), max(v, u)) in edges for u in current):
                    # New candidates are those after v that connect to v
                    new_candidates = [u for u in candidates[i+1:] if (min(v, u), max(v, u)) in edges]
                    expand(current + [v], new_candidates)
        
        expand([], sorted_verts)
        return best_clique

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        remaining_edges = set(edges)
        
        cliques = []
        
        # Process each vertex in I (independent set)
        for v in range(k, n):
            # Find neighbors of v in C that are still available (edge not covered)
            S_avail = []
            for u in range(k):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    S_avail.append(u)
            
            if not S_avail:
                continue
            
            # Partition S_avail into cliques of the remaining graph
            # We want to minimize number of cliques, so we make them as large as possible
            while S_avail:
                # Find a maximum clique in the subgraph induced by S_avail
                clique_part = max_clique(S_avail, remaining_edges)
                if not clique_part:
                    # Should not happen if edges exist, but safety break
                    break
                
                # The full clique is {v} U clique_part
                full_clique = [v] + clique_part
                cliques.append(full_clique)
                
                # Remove all edges in this clique from remaining_edges
                for i in range(len(full_clique)):
                    for j in range(i + 1, len(full_clique)):
                        u, w = full_clique[i], full_clique[j]
                        e = (u, w) if u < w else (w, u)
                        remaining_edges.discard(e)
                
                # Remove clique_part from S_avail
                S_avail = [u for u in S_avail if u not in clique_part]
        
        # Process remaining edges (all in C)
        # Vertices are 0..k-1
        remaining_vertices = set(range(k))
        
        while remaining_edges:
            # Find a maximal clique in the remaining graph
            # We can just pick a vertex and expand
            # Or use the helper function on the set of vertices that have remaining edges
            
            # Get vertices involved in remaining edges
            verts_involved = set()
            for u, w in remaining_edges:
                verts_involved.add(u)
                verts_involved.add(w)
            
            if not verts_involved:
                break
            
            clique_part = max_clique(list(verts_involved), remaining_edges)
            if not clique_part or len(clique_part) < 2:
                # Fallback: take any edge
                u, w = next(iter(remaining_edges))
                clique_part = [u, w]
            
            cliques.append(clique_part)
            
            # Remove edges
            for i in range(len(clique_part)):
                for j in range(i + 1, len(clique_part)):
                    u, w = clique_part[i], clique_part[j]
                    e = (u, w) if u < w else (w, u)
                    remaining_edges.discard(e)
            
            # If we didn't remove all edges, loop continues
            # But we must ensure progress.
            # If clique_part is size 1, we didn't remove any edge.
            # But find_maximal_clique returns at least 1 vertex.
            # If size 1, no edges removed.
            # We should ensure we pick at least 2 vertices if possible.
            # If remaining_edges is not empty, there is at least one edge (u, w).
            # So verts_involved has at least u, w.
            # find_maximal_clique should return at least {u, w} if (u, w) in edges.
            # So size >= 2.
            
        results.append(cliques)
    
    return results
