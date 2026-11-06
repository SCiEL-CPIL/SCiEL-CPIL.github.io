    for _ in range(5000):
        if time.time() > deadline: break
        
        existing_edges = list(edges)
        if not existing_edges: break
        
        ra, rb = random.choice(existing_edges)
        
        # Deep copy adj and edges
        adj_copy = [row[:] for row in adj]
        edges_copy = set(edges)
        
        # Remove
        adj[ra][rb] = -1
        adj[rb][ra] = -1
        edges.remove((ra, rb))
        
        # Greedy add pass
        missing = [(u,v) for u,v in pairs if (u,v) not in edges and (v,u) not in edges]
        random.shuffle(missing)
        for u, v in missing:
            try_add(u, v)
            
        if len(edges) > len(edges_copy):
            # Keep
            pass
        else:
            # Revert
            adj = adj_copy
            edges = edges_copy
