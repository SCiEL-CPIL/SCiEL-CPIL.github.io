<<<<<<< SEARCH
    # Fallback
    if best_edges == 0:
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 80

    # Phase 2: Local Search
    # We improve the local search by:
    # 1. Keeping track of the best solution found during local search
    # 2. Using a more aggressive escape mechanism (removing edges) when stuck
    # 3. Trying to add edges in a more systematic way (not just random, but also checking all non-edges if time permits)
    
    adj = best_adj[:]
    current_edges = best_edges
    best_local_edges = best_edges
    best_local_adj = best_adj[:]
    
    no_improve_count = 0
    
    while time_left() > 2:
        added = False
        
        # Try to add a random non-edge
        # We try more candidates per iteration to increase the chance of finding an addable edge
        for _ in range(100):
            if time_left() < 2:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve_count = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                break
        
        if not added:
            # Escape local maximum by removing an edge
            # We lower the threshold for removing edges to escape local maxima more aggressively
            if no_improve_count > 10:
                # Find a random edge to remove
                # Collect all edges for random selection
                edge_list = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        mask &= mask - 1
                        if i < j:
                            edge_list.append((i, j))
                
                if edge_list:
                    u, v = random.choice(edge_list)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                    no_improve_count = 0
                else:
                    break
            
            no_improve_count += 1

    if best_local_edges > best_edges:
        best_edges = best_local_edges
        best_adj = best_local_adj

    return get_edges()
=======
    # Fallback
    if best_edges == 0:
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 80

    # Phase 2: Local Search with 1-edge-swap
    adj = best_adj[:]
    current_edges = best_edges
    best_local_edges = best_edges
    best_local_adj = best_adj[:]
    
    no_improve_count = 0
    swap_count = 0
    
    while time_left() > 3:
        added = False
        
        # Try to add a random non-edge
        for _ in range(50):
            if time_left() < 3:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve_count = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                break
        
        if not added:
            # Systematic sweep: try all non-edges
            if no_improve_count % 5 == 0 and time_left() > 10:
                for u, v in all_pairs:
                    if time_left() < 3:
                        break
                    if adj[u] & (1 << v):
                        continue
                    if can_add(u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                        added = True
                        no_improve_count = 0
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_adj = adj[:]
                        break
            
            if not added:
                # 1-edge-swap: remove one edge, try to add a different one
                if no_improve_count > 5:
                    # Collect all edges
                    edge_list = []
                    for i in range(n):
                        mask = adj[i]
                        while mask:
                            j = (mask & -mask).bit_length() - 1
                            mask &= mask - 1
                            if i < j:
                                edge_list.append((i, j))
                    
                    if edge_list:
                        # Try several swaps
                        swap_success = False
                        for _ in range(20):
                            if time_left() < 3:
                                break
                            u, v = random.choice(edge_list)
                            # Remove edge (u,v)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            current_edges -= 1
                            
                            # Try to add a random non-edge
                            found = False
                            for _2 in range(30):
                                if time_left() < 3:
                                    break
                                a, b = random.choice(all_pairs)
                                if adj[a] & (1 << b):
                                    continue
                                if (a == u and b == v) or (a == v and b == u):
                                    continue
                                if can_add(a, b):
                                    adj[a] |= (1 << b)
                                    adj[b] |= (1 << a)
                                    current_edges += 1
                                    found = True
                                    swap_count += 1
                                    if current_edges > best_local_edges:
                                        best_local_edges = current_edges
                                        best_local_adj = adj[:]
                                    break
                            
                            # If we couldn't add an edge, restore the removed one
                            if not found:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                current_edges += 1
                            
                            if found:
                                swap_success = True
                                no_improve_count = 0
                                break
                        if swap_success:
                            continue
                    
                    no_improve_count += 1
                else:
                    no_improve_count += 1

    if best_local_edges > best_edges:
        best_edges = best_local_edges
        best_adj = best_local_adj

    return get_edges()
>>>>>>> REPLACE
