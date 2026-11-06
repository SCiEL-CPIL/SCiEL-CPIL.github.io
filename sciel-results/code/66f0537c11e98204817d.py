import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def andrasfai_graph(n):
        """Andrásfai graph A_k where n = 3k, k >= 4.
        Triangle-free, alpha = k."""
        k = n // 3
        if 3 * k != n:
            return None
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    for j in range(k):
                        if (b - a) % 3 == 1 and (j == i or (j - i) % k == 1):
                            u = a * k + i
                            v = b * k + j
                            if u < v:
                                edges.append([u, v])
        return {'n': n, 'edges': edges}

    def mycielski(n, edges):
        """Apply Mycielski construction to a graph with n vertices and given edges.
        Returns (new_n, new_edges)."""
        new_n = 2 * n + 1
        new_edges = list(edges)
        
        # Copy vertices n..2n-1
        # For each original edge (u, v), add (u, v+n) and (v, u+n)
        # Wait, the definition is:
        # v'_i is connected to N(v_i).
        # So for each edge (u, v) in G, add (u, v') and (v, u')?
        # No, v'_i is connected to all neighbors of v_i.
        # So if (u, v) is an edge, then u is a neighbor of v.
        # So v'_v is connected to u.
        # And u'_u is connected to v.
        
        # Let's iterate neighbors.
        # Build adjacency list for original graph
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for i in range(n):
            for neighbor in adj[i]:
                # Add edge (neighbor, i+n)
                new_edges.append([neighbor, i + n])
                
        # Add root vertex 2n
        # Connect root to all v'_i (which are i+n)
        for i in range(n):
            new_edges.append([2 * n, i + n])
            
        return new_n, new_edges

    # Step 1: Construct A_6 (n=18)
    base = andrasfai_graph(18)
    if base is None:
        # Fallback to A_9 if something goes wrong, though 18 is valid
        base = andrasfai_graph(27)
        if base is None:
            return {'n': 1, 'edges': []}
        
    n_base = base['n']
    edges_base = base['edges']
    
    # Step 2: Add one isolated vertex to A_6
    # New n = 19
    n_mid = n_base + 1
    edges_mid = list(edges_base)
    # Vertex 18 is isolated, so no edges added
    
    # Step 3: Mycielski Construction
    n_final, edges_final = mycielski(n_mid, edges_mid)
    
    # Verify time
    if time.time() - start_time > deadline:
         # This shouldn't happen as construction is fast
         pass
         
    return {'n': n_final, 'edges': edges_final}
