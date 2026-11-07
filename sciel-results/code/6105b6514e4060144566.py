def construct():
    from random import Random
    import sys
    
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

    def find_max_clique(vertices, edges):
        """
        Find a maximum (or near-maximum) clique in the subgraph induced by 'vertices'.
        'vertices' is a list of ints. 'edges' is a set of tuples.
        """
        if not vertices:
            return []
        
        # Convert to set for fast lookup
        v_set = set(vertices)
        
        # Build adjacency list within vertices
        adj = {v: set() for v in vertices}
        for u in vertices:
            for w in vertices:
                if u < w and (u, w) in edges:
                    adj[u].add(w)
                    adj[w].add(u)
        
        # Greedy approach with sorting by degree
        # This is a heuristic for maximum clique
        # Sort by degree descending
        sorted_verts = sorted(vertices, key=lambda x: -len(adj[x]))
        
        clique = []
        candidates = set(vertices)
        
        # Simple greedy: pick highest degree, add to clique, remove non-neighbors
        # This finds a maximal clique. To improve, we can try to maximize size.
        # For small k, we can use a better method.
        # Let's use a recursive branch and bound for k <= 30?
        # Or just a good greedy.
        
        # Let's try a simple greedy that picks the vertex with highest degree in the remaining candidates
        # that is connected to all in current clique.
        
        current_clique = []
        candidates = set(vertices)
        
        while candidates:
            # Find vertex in candidates with highest degree in the subgraph induced by candidates
            # that is connected to all in current_clique
            best_v = None
            best_deg = -1
            for v in candidates:
                if all((min(v, u), max(v, u)) in edges for u in current_clique):
                    # Count neighbors in candidates
                    deg = sum(1 for w in candidates if w != v and (min(v, w), max(v, w)) in edges)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            
            if best_v is None:
                break
            
            current_clique.append(best_v)
            # Remove best_v and its non-neighbors from candidates
            candidates.discard(best_v)
            non_neighbors = {w for w in candidates if (min(best_v, w), max(best_v, w)) not in edges}
            candidates -= non_neighbors
            
        return current_clique

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        remaining_edges = set(edges)
        
        cliques = []
        
        # Prepare list of I vertices and their neighbors in C
        i_vertices = []
        for v in range(k, n):
            N_v = []
            for u in range(k):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    N_v.append(u)
            if N_v:
                i_vertices.append((v, N_v))
        
        # Sort by size of N_v descending
        i_vertices.sort(key=lambda x: -len(x[1]))
        
        for v, N_v in i_vertices:
            S_avail = list(N_v)
            while S_avail:
                # Find max clique in S_avail
                clique_part = find_max_clique(S_avail, remaining_edges)
                if not clique_part:
                    # Safety break, should not happen if edges exist
                    break
                
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
        while remaining_edges:
            verts_involved = set()
            for u, w in remaining_edges:
                verts_involved.add(u)
                verts_involved.add(w)
            
            if not verts_involved:
                break
            
            clique_part = find_max_clique(list(verts_involved), remaining_edges)
            if not clique_part or len(clique_part) < 2:
                # Fallback
                u, w = next(iter(remaining_edges))
                clique_part = [u, w]
            
            cliques.append(clique_part)
            
            for i in range(len(clique_part)):
                for j in range(i + 1, len(clique_part)):
                    u, w = clique_part[i], clique_part[j]
                    e = (u, w) if u < w else (w, u)
                    remaining_edges.discard(e)
            
        results.append(cliques)
    
    return results
