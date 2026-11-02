<<<<<<< SEARCH
    # Strategy 1: Multiple greedy with different orderings
    # Key insight: vectors that are "more compatible" with many others should be picked first
    # Compute for each vector, how many others it's compatible with (dot <= 2)
    
    # compat_count[i] = number of j ≠ i such that M[i,j] <= 2
    # Exclude self: M[i,i] = 4 > 2, so we need to be careful
    compat = (M <= 2).astype(np.int32)
    # Subtract self-compatibility (M[i,i]=4, so compat[i,i]=0 already since 4>2)
    # Actually M[i,i] = 4, and 4 <= 2 is False, so compat[i,i] = 0. Good.
    compat_count = compat.sum(axis=1)
    
    # Sort by descending compatibility count (most compatible first)
    order_comp = np.argsort(-compat_count)
    
    orderings = [order_comp, np.arange(N)]
    # Add some random permutations
    for _ in range(50):
        orderings.append(rng.permutation(N))
    
    # Also try: sort by ascending index within groups of same type
    # Type 1: indices 0..21, Type 2: indices 22..N-1
    # For type 2, group by which 4 coordinates are used
    
    for oi, order in enumerate(orderings):
        if time.time() > deadline:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            # Check compatibility with all selected
            # M[idx, selected_mask] gives dot products with selected vectors
            # We need all to be <= 2
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            # If we hit 512, we're done
            if best_size >= 512:
                break
    
    # Local search: try to swap elements to improve
    if time.time() < deadline - 5:
        if best_size > 0:
            # Convert to set for faster lookup
            best_set_arr = np.array(best_set)
            in_set = np.zeros(N, dtype=bool)
            in_set[best_set_arr] = True
            
            # Try single swaps: remove one, add another
            # For each selected element i, find unselected j that is compatible with all except i
            # i.e., for all k in selected \ {i}, M[j,k] <= 2, and M[j,i] can be anything
            # Actually we need j compatible with selected \ {i}
            
            # Precompute: for each unselected j, which selected elements is it incompatible with?
            # incomp[j] = set of selected indices where M[j, selected_idx] > 2
            
            # This is expensive. Let's do it more cleverly.
            # For a swap to work: remove i, add j. j must be compatible with all selected except possibly i.
            # So count_incompat[j] = number of selected k where M[j,k] > 2. We need count_incompat[j] <= 1.
            # And if count_incompat[j] == 1, that one must be i.
            
            if time.time() < deadline - 5:
                # Compute incompatibility counts for unselected vectors
                # M[j, in_set] for j not in set
                # This is a large matrix op but let's try
                if N - best_size < 5000:  # only if not too many unselected
                    unselected_indices = np.where(~in_set)[0]
                    if len(unselected_indices) > 0:
                        # For each unselected j, count how many selected it's incompatible with
                        # M[unselected_indices, :][:, in_set] shape (n_unsel, best_size)
                        sub_M = M[np.ix_(unselected_indices, best_set_arr)]
                        incomp_counts = (sub_M > 2).sum(axis=1)
                        
                        # Find candidates where incomp_counts <= 1
                        swap_candidates = unselected_indices[incomp_counts <= 1]
                        
                        # For each candidate j, try to find an i to remove
                        # If incomp_counts[j] == 0: can swap with any i (but no improvement in size)
                        # If incomp_counts[j] == 1: must remove the specific i that's incompatible
                        # This doesn't increase size, it's for diversity or future improvements
                        # Actually for maximizing size, we need ADDITIONS not swaps
                        # Let me try a different local search: try adding vectors that are almost compatible
                        
                        # Try to find vectors that are incompatible with only 1 selected, 
                        # then see if we can remove that one and add two others
                        # This is complex. Let's skip for now.
                        pass
=======
    # Strategy: Multiple greedy with different orderings + local search
    # Key insight: vectors that are "more compatible" with many others should be picked first
    
    # compat_count[i] = number of j ≠ i such that M[i,j] <= 2
    compat = (M <= 2).astype(np.int32)
    compat_count = compat.sum(axis=1)
    
    # Sort by descending compatibility count (most compatible first)
    order_comp = np.argsort(-compat_count)
    
    # Generate diverse orderings
    orderings = [order_comp, np.arange(N)]
    
    # Add random permutations
    for _ in range(30):
        orderings.append(rng.permutation(N))
    
    # Add orderings based on type (Type 1 first, then Type 2 by coordinate pattern)
    type1_indices = np.arange(22)
    type2_indices = np.arange(22, N)
    orderings.append(np.concatenate([type1_indices, type2_indices]))
    orderings.append(np.concatenate([type2_indices, type1_indices]))
    
    # Shuffle within types
    for _ in range(10):
        t1_shuffled = type1_indices.copy()
        t2_shuffled = type2_indices.copy()
        rng.shuffle(t1_shuffled)
        rng.shuffle(t2_shuffled)
        orderings.append(np.concatenate([t1_shuffled, t2_shuffled]))
    
    # Greedy with max-min heuristic: at each step, pick the vector that maximizes 
    # the minimum compatibility with remaining candidates
    def greedy_max_min(order, deadline):
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        remaining = set(order.tolist())
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_idx = None
            best_score = -1
            
            for idx in remaining:
                if selected:
                    # Score = minimum dot product with selected (want to maximize this, 
                    # but it must be <= 2 for validity)
                    # Actually we want vectors that are compatible (dot <= 2) and 
                    # have high minimum dot product (closer to 2 means more "central")
                    dots = M[idx, selected_mask]
                    if np.all(dots <= 2):
                        # Compatible - score by how "central" it is
                        score = np.min(dots)
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                else:
                    best_idx = idx
                    break
            
            if best_idx is None:
                break
            
            selected.append(best_idx)
            selected_mask[best_idx] = True
            remaining.remove(best_idx)
        
        return selected
    
    # Run all orderings
    for oi, order in enumerate(orderings):
        if time.time() > deadline:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size >= 512:
                break
    
    # Local search: try to improve by finding vectors that can be added
    # by removing a small number of selected vectors
    if time.time() < deadline - 10 and best_size > 0:
        best_set_arr = np.array(best_set)
        in_set = np.zeros(N, dtype=bool)
        in_set[best_set_arr] = True
        
        # Compute incompatibility matrix for unselected vs selected
        unselected_indices = np.where(~in_set)[0]
        if len(unselected_indices) > 0:
            sub_M = M[np.ix_(unselected_indices, best_set_arr)]
            incomp_counts = (sub_M > 2).sum(axis=1)
            
            # Try to add vectors that are incompatible with at most 2 selected
            # For each such vector, try removing the incompatible ones and adding it
            # plus potentially other vectors that become compatible
            
            # First pass: just try adding vectors with incomp_counts == 0 (shouldn't exist if greedy was optimal)
            # Second pass: try 1-removal swaps (remove 1, add 1 - no size gain)
            # Third pass: try 2-removal, 3-addition (size gain of 1)
            
            # For efficiency, focus on vectors with incomp_counts <= 2
            candidate_mask = incomp_counts <= 2
            candidates = unselected_indices[candidate_mask]
            
            if len(candidates) > 0 and time.time() < deadline - 5:
                # For each candidate, get which selected it's incompatible with
                cand_incomp = (sub_M[candidate_mask] > 2)  # shape (n_cand, best_size)
                
                # Try to find a set of candidates that can be added by removing few selected
                # Simple approach: for each candidate with incomp_count == 1, 
                # check if we can also add another candidate that's compatible with the rest
                for ci in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    
                    j = candidates[ci]
                    incomp_sel = np.where(cand_incomp[ci])[0]  # indices in best_set_arr that are incompatible
                    
                    if len(incomp_sel) == 0:
                        # Can add directly (shouldn't happen if greedy was perfect)
                        if not in_set[j]:
                            best_set.append(j)
                            in_set[j] = True
                            best_size += 1
                    elif len(incomp_sel) == 1:
                        # Remove one selected, add this one - no size gain
                        # But maybe we can add two by removing one
                        remove_idx = incomp_sel[0]
                        # Check if any other candidate is compatible with all selected except remove_idx
                        # and also compatible with j
                        for cj in range(ci + 1, len(candidates)):
                            if time.time() > deadline:
                                break
                            k = candidates[cj]
                            incomp_sel_k = np.where(cand_incomp[cj])[0]
                            # k must be compatible with all selected except possibly remove_idx
                            if len(incomp_sel_k) <= 1:
                                # Check if k is compatible with j
                                if M[j, k] <= 2:
                                    # Check if k is compatible with all selected except remove_idx
                                    ok = True
                                    for ri in incomp_sel_k:
                                        if ri != remove_idx:
                                            ok = False
                                            break
                                    if ok:
                                        # Remove best_set_arr[remove_idx], add j and k
                                        old_idx = best_set_arr[remove_idx]
                                        in_set[old_idx] = False
                                        in_set[j] = True
                                        in_set[k] = True
                                        # Update best_set
                                        new_best_set = [x for x in best_set if x != old_idx]
                                        new_best_set.extend([j, k])
                                        best_set = new_best_set
                                        best_set_arr = np.array(best_set)
                                        best_size = len(best_set)
                                        # Recompute in_set
                                        in_set = np.zeros(N, dtype=bool)
                                        in_set[best_set_arr] = True
                                        # Recompute sub_M and incomp_counts for next iteration
                                        unselected_indices = np.where(~in_set)[0]
                                        if len(unselected_indices) == 0:
                                            break
                                        sub_M = M[np.ix_(unselected_indices, best_set_arr)]
                                        incomp_counts = (sub_M > 2).sum(axis=1)
                                        candidate_mask = incomp_counts <= 2
                                        candidates = unselected_indices[candidate_mask]
                                        cand_incomp = (sub_M[candidate_mask] > 2)
                                        break
>>>>>>> REPLACE

