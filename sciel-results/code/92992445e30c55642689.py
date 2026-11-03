<<<<<<< SEARCH
    # Find Maximum Independent Set (MIS) of this graph
    best_set = []
    best_size = 0
    
    # Precompute degrees for initial greedy
    degrees = np.sum(conflict_supports, axis=1)
    
    # Greedy MIS with random restarts and local search
    num_restarts = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Randomize order slightly or use degree-based
        # Strategy: Sort by degree, but add random noise to break ties or perturb order
        # Or just random permutation
        order = np.random.permutation(N_S)
        
        selected_mask = np.zeros(N_S, dtype=bool)
        selected = []
        
        # Greedy Phase
        for idx in order:
            if selected_mask[idx]:
                continue
            # Check if idx conflicts with any selected
            # conflict_supports[idx] is a boolean array
            if not np.any(conflict_supports[idx] & selected_mask):
                selected.append(idx)
                selected_mask[idx] = True
                
        # Local Search Phase: 1-for-2 swaps
        improved = True
        while improved:
            improved = False
            # Try to add any node
            for idx in range(N_S):
                if selected_mask[idx]:
                    continue
                if not np.any(conflict_supports[idx] & selected_mask):
                    selected.append(idx)
                    selected_mask[idx] = True
                    improved = True
            
            # Try 1-for-2 swaps
            # For each u in selected, check if we can remove u and add two others
            # This is O(N^3) worst case, but N=330, so 330^3 ~ 35M. Might be slow in Python loop.
            # Optimize: Only try swaps if size is small or use numpy.
            # Given time limit, we might skip complex swaps if greedy is good enough.
            # However, 1-for-2 is powerful.
            # Let's try a simpler local search: check all unselected nodes against all selected nodes?
            # The "add any node" loop above is O(N^2).
            # The swap loop is expensive.
            # Let's stick to "add any node" and maybe "remove one, add two" if time permits.
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected.copy()
=======
    # Find Maximum Independent Set (MIS) of this graph
    best_set = []
    best_size = 0
    
    # Precompute degrees for initial greedy
    degrees = np.sum(conflict_supports, axis=1)
    
    # Greedy MIS with random restarts and local search
    num_restarts = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Randomize order slightly or use degree-based
        # Strategy: Sort by degree, but add random noise to break ties or perturb order
        # Or just random permutation
        order = np.random.permutation(N_S)
        
        selected_mask = np.zeros(N_S, dtype=bool)
        selected = []
        
        # Greedy Phase
        for idx in order:
            if selected_mask[idx]:
                continue
            # Check if idx conflicts with any selected
            # conflict_supports[idx] is a boolean array
            if not np.any(conflict_supports[idx] & selected_mask):
                selected.append(idx)
                selected_mask[idx] = True
                
        # Local Search Phase: add nodes and 1-for-2 swaps
        improved = True
        while improved:
            improved = False
            
            # Try to add any node
            for idx in range(N_S):
                if selected_mask[idx]:
                    continue
                if not np.any(conflict_supports[idx] & selected_mask):
                    selected.append(idx)
                    selected_mask[idx] = True
                    improved = True
            
            # Try 1-for-2 swaps: remove one selected, add two unselected
            # For efficiency, only try this if we have time
            if time.time() < deadline - 10:
                selected_list = selected.copy()
                for u_pos in range(len(selected_list)):
                    u = selected_list[u_pos]
                    # Try removing u
                    temp_mask = selected_mask.copy()
                    temp_mask[u] = False
                    
                    # Find all nodes that can be added if u is removed
                    # A node v can be added if: not selected, and doesn't conflict with any selected except possibly u
                    addable = []
                    for v in range(N_S):
                        if temp_mask[v]:
                            continue
                        # v must not conflict with any currently selected (after removing u)
                        if not np.any(conflict_supports[v] & temp_mask):
                            addable.append(v)
                    
                    # Try to find two addable nodes that don't conflict with each other
                    # and also don't conflict with u (since we're removing u, they just need to be compatible with remaining)
                    # Actually, they need to be compatible with all remaining selected
                    # temp_mask already excludes u, so we just need v1 and v2 to be compatible with each other
                    # and both compatible with temp_mask (which we already checked)
                    
                    # Find pairs in addable that don't conflict
                    found_pair = False
                    for i in range(len(addable)):
                        v1 = addable[i]
                        for j in range(i + 1, len(addable)):
                            v2 = addable[j]
                            if not conflict_supports[v1, v2]:
                                # Found a valid swap
                                selected_mask[u] = False
                                selected.remove(u)
                                selected.append(v1)
                                selected.append(v2)
                                selected_mask[v1] = True
                                selected_mask[v2] = True
                                improved = True
                                found_pair = True
                                break
                        if found_pair:
                            break
                    
                    if found_pair:
                        # Restart the outer while loop since we made a change
                        break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
>>>>>>> REPLACE

