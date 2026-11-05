    def construct():
        # Start with K2
        n = 2
        edges = [(0, 1)]
        
        # Apply Mycielski 4 times to get n=47
        for _ in range(4):
            new_edges = []
            # 1. Copy original edges
            for u, v in edges:
                new_edges.append((u, v))
            
            # 2. Add edges from clones to original neighbors
            # For each original edge (u, v):
            # u' (n+u) is connected to v
            # v' (n+v) is connected to u
            for u, v in edges:
                new_edges.append((n + u, v))
                new_edges.append((n + v, u))
            
            # 3. Add edges from apex to all clones
            apex = 2 * n
            for u in range(n):
                new_edges.append((apex, n + u))
            
            edges = new_edges
            n = 2 * n + 1
        
        # Now n=47, edges is the list of edges for G4
        # We want n_target = 40
        n_target = 40
        final_edges = []
        for u, v in edges:
            if u < n_target and v < n_target:
                final_edges.append([u, v])
        
        return {'n': n_target, 'edges': final_edges}
    