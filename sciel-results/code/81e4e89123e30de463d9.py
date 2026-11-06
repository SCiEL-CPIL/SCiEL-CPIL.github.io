<<<<<<< SEARCH
    # Extension loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 10.0:
            break
        
        # Time budget per vertex
        time_per_vertex = 5.0
        if target_n > 40:
            time_per_vertex = 10.0 # Give more time for harder steps
        
        found_N = find_valid_N(n, k4_list, alpha4_list, time_per_vertex)
        
        if found_N is None:
            # Try a few more times with different random seeds or longer time?
            # If we fail, we might be stuck.
            # We can try to modify the existing graph? Too complex.
            # Just break.
            break
        
        # Add the new vertex
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update k4_list and alpha4_list
        # New K4s: Any K3 in the new vertex's neighborhood + the new vertex
        # New Alpha4s: Any Alpha3 in the new vertex's non-neighborhood + the new vertex
        
        # Get K3s in adj[new_vertex]
        new_k4s = []
        neigh_mask = adj[new_vertex]
        # Iterate over triples in neigh_mask
        # Efficient way: iterate a < b < c in neigh_mask
        # We can just iterate all triples in 0..n-2 and check if in neigh_mask
        # n is small, so O(n^3) is fine.
        for a in range(n - 1):
            if not (neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (neigh_mask >> b) & 1:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                        new_k4s.append(mask)
        
        k4_list.extend(new_k4s)
        
        # Get Alpha3s in ~neigh_mask (within 0..n-2)
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1: # Edge exists, so not independent
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
        
        alpha4_list.extend(new_alpha4s)
        
        # Optional: Prune lists? No, just keep them.
=======
    # Extension loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 10.0:
            break
        
        # Time budget per vertex
        # Start with 5s, increase for larger n
        time_per_vertex = 5.0
        if target_n > 40:
            time_per_vertex = 10.0
        if target_n > 42:
            time_per_vertex = 20.0 # Harder steps
        
        found_N = find_valid_N(n, k4_list, alpha4_list, time_per_vertex)
        
        if found_N is None:
            # If we fail to find a valid N, we can't extend.
            # We could try to "repair" the graph by flipping edges, but that's complex.
            # For now, we stop.
            break
        
        # Add the new vertex
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update k4_list and alpha4_list
        # New K4s: Any K3 in the new vertex's neighborhood + the new vertex
        # New Alpha4s: Any Alpha3 in the new vertex's non-neighborhood + the new vertex
        
        # Get K3s in adj[new_vertex]
        new_k4s = []
        neigh_mask = adj[new_vertex]
        # Iterate over triples in neigh_mask
        for a in range(n - 1):
            if not (neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (neigh_mask >> b) & 1:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                        new_k4s.append(mask)
        
        k4_list.extend(new_k4s)
        
        # Get Alpha3s in ~neigh_mask (within 0..n-2)
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1: # Edge exists, so not independent
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
        
        alpha4_list.extend(new_alpha4s)
        
        # Update numpy arrays for the new lists
        # This is O(K*n) which is fine for small K
        if new_k4s:
            new_k4_arr = np.array([[1 if (mask >> i) & 1 else 0 for i in range(n)] for mask in new_k4s], dtype=np.uint8)
            k4_arr = np.vstack([k4_arr, new_k4_arr])
        if new_alpha4s:
            new_alpha4_arr = np.array([[1 if (mask >> i) & 1 else 0 for i in range(n)] for mask in new_alpha4s], dtype=np.uint8)
            alpha4_arr = np.vstack([alpha4_arr, new_alpha4_arr])
>>>>>>> REPLACE
