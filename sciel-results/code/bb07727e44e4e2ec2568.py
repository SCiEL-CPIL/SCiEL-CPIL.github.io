<<<<<<< SEARCH
    # Try Tomita branch-and-bound with time limit
    # Use remaining time for B&B
    remaining_time = deadline - time.time() - 5  # 5s safety margin
    
    if remaining_time > 2:
        try:
            # Run Tomita with the remaining time
            # To make it faster, we can use the greedy solution as a lower bound
            # and only explore branches that could beat it
            
            # For now, just run Tomita and see if it improves
            # Note: 5280 vertices might be too large for pure Python Tomita in 125s
            # Let's try with a smaller effective search by using the greedy solution as starting point
            
            # Alternative: use a faster implementation with numpy
            # Let's try a different approach: local search on top of greedy
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                selected_arr = np.array(best_selected)
                selected_set = set(best_selected)
                
                # Try to add vertices that don't conflict with selected
                # Find all vertices that don't conflict with any selected
                conflict_with_selected = np.zeros(N4, dtype=bool)
                for idx in selected_arr:
                    conflict_with_selected |= conflicts[idx]
                
                # Candidates: not selected, not conflicting with selected
                candidates = np.where(~conflict_with_selected & ~np.isin(np.arange(N4), selected_arr))[0]
                
                if len(candidates) == 0:
                    break
                
                # Try to find an independent set of size >= 2 from candidates
                # Greedy approach
                cands_list = candidates.tolist()
                rng.shuffle(cands_list)
                
                best_addition = []
                for i, cand1 in enumerate(cands_list):
                    if len(best_addition) >= 2:
                        break
                    # Find another candidate that doesn't conflict with cand1
                    for cand2 in cands_list[i+1:]:
                        if not conflicts[cand1, cand2]:
                            best_addition = [cand1, cand2]
                            break
                    if len(best_addition) >= 2:
                        break
                
                if len(best_addition) >= 2:
                    for cand in best_addition:
                        best_selected.append(cand)
                    improved = True
                    best_size = len(best_selected)
            
            # Try swap-based local search
            if time.time() < deadline - 5:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    selected_arr = np.array(best_selected)
                    
                    for i in range(len(selected_arr)):
                        if time.time() > deadline - 5:
                            break
                        remove_idx = selected_arr[i]
                        remaining = np.setdiff1d(selected_arr, [remove_idx])
                        
                        # Find candidates that don't conflict with remaining
                        conflict_with_remaining = np.zeros(N4, dtype=bool)
                        for idx in remaining:
                            conflict_with_remaining |= conflicts[idx]
                        
                        candidates = np.where(~conflict_with_remaining & ~np.isin(np.arange(N4), selected_arr))[0]
                        
                        # Try to add 2+ non-conflicting candidates
                        cands_list = candidates.tolist()
                        rng.shuffle(cands_list)
                        
                        added = []
                        for cand in cands_list:
                            if len(added) >= 2:
                                break
                            ok = True
                            for a in added:
                                if conflicts[cand, a]:
                                    ok = False
                                    break
                            if ok:
                                added.append(cand)
                        
                        if len(added) >= 2:
                            # Perform swap
                            new_selected = list(selected_arr)
                            new_selected.pop(i)
                            new_selected.extend(added)
                            best_selected = new_selected
                            improved = True
                            best_size = len(best_selected)
                            break
            
        except Exception:
            # If B&B or local search fails, keep the greedy solution
            pass
=======
    # Vectorized local search to improve the independent set
    if time.time() < deadline - 5:
        try:
            # Precompute: for each vertex, which other vertices conflict with it
            # conflicts is already computed as a boolean matrix
            
            # Strategy 1: Greedy addition - repeatedly add vertices that don't conflict with current set
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                selected_arr = np.array(best_selected, dtype=np.int64)
                if len(selected_arr) == 0:
                    break
                    
                # Find all vertices that conflict with any selected vertex
                # Vectorized: OR of conflict rows for selected vertices
                conflict_mask = np.zeros(N4, dtype=bool)
                if len(selected_arr) > 0:
                    conflict_mask[selected_arr] = True  # Mark selected as conflicting with themselves
                    # OR together conflict rows
                    conflict_mask = np.any(conflicts[selected_arr], axis=0)
                
                # Candidates: not conflicting with any selected
                candidates = np.where(~conflict_mask)[0]
                
                if len(candidates) == 0:
                    break
                
                # Greedily add candidates that don't conflict with each other
                # Process in order of lowest degree within candidates for better packing
                cand_indices = candidates
                cand_set = set(candidates.tolist())
                
                # Compute degrees within candidate set
                cand_degrees = np.zeros(len(candidates), dtype=np.int32)
                for i, c in enumerate(candidates):
                    # Count neighbors of c that are also in candidates
                    neigh = np.where(conflicts[c])[0]
                    in_cand = np.isin(neigh, candidates)
                    cand_degrees[i] = np.sum(in_cand)
                
                # Sort by degree (ascending) - prefer low-degree vertices
                order = np.argsort(cand_degrees)
                sorted_candidates = candidates[order]
                
                added_this_round = []
                current_conflicts = conflict_mask.copy()
                
                for cand in sorted_candidates:
                    if time.time() > deadline - 5:
                        break
                    # Check if cand conflicts with any already added in this round
                    if not current_conflicts[cand]:
                        added_this_round.append(cand)
                        # Update conflict mask with this new vertex's conflicts
                        current_conflicts[cand] = True
                        current_conflicts[np.where(conflicts[cand])[0]] = True
                
                if len(added_this_round) > 0:
                    best_selected.extend(added_this_round)
                    improved = True
                    best_size = len(best_selected)
            
            # Strategy 2: Swap-based local search - remove one, add two or more
            if time.time() < deadline - 5:
                improved = True
                max_swaps = 50
                swap_count = 0
                while improved and time.time() < deadline - 5 and swap_count < max_swaps:
                    improved = False
                    selected_arr = np.array(best_selected, dtype=np.int64)
                    n_sel = len(selected_arr)
                    
                    # Try removing each selected vertex and see if we can add more
                    # Use random order to avoid bias
                    perm = rng.permutation(n_sel)
                    
                    for idx in perm:
                        if time.time() > deadline - 5:
                            break
                        
                        remove_idx = selected_arr[idx]
                        remaining = np.setdiff1d(selected_arr, [remove_idx])
                        
                        # Find candidates that don't conflict with remaining
                        if len(remaining) == 0:
                            conflict_mask = np.zeros(N4, dtype=bool)
                        else:
                            conflict_mask = np.any(conflicts[remaining], axis=0)
                        
                        # Also exclude all currently selected (we'll re-add remove_idx if beneficial)
                        conflict_mask[selected_arr] = True
                        
                        candidates = np.where(~conflict_mask)[0]
                        
                        if len(candidates) < 2:
                            continue
                        
                        # Try to add multiple non-conflicting candidates
                        cand_indices = candidates
                        cand_set = set(candidates.tolist())
                        
                        # Compute degrees within candidate set
                        cand_degrees = np.zeros(len(candidates), dtype=np.int32)
                        for i, c in enumerate(candidates):
                            neigh = np.where(conflicts[c])[0]
                            in_cand = np.isin(neigh, candidates)
                            cand_degrees[i] = np.sum(in_cand)
                        
                        order = np.argsort(cand_degrees)
                        sorted_candidates = candidates[order]
                        
                        added_this_swap = []
                        current_conflicts = conflict_mask.copy()
                        
                        for cand in sorted_candidates:
                            if time.time() > deadline - 5:
                                break
                            if len(added_this_swap) >= 3:  # Try to add up to 3
                                break
                            if not current_conflicts[cand]:
                                added_this_swap.append(cand)
                                current_conflicts[cand] = True
                                current_conflicts[np.where(conflicts[cand])[0]] = True
                        
                        if len(added_this_swap) >= 2:
                            # Perform swap: remove one, add two or more
                            new_selected = list(selected_arr)
                            new_selected.pop(idx)
                            new_selected.extend(added_this_swap)
                            best_selected = new_selected
                            improved = True
                            best_size = len(best_selected)
                            swap_count += 1
                            break  # Restart with new selection
            
        except Exception:
            pass
>>>>>>> REPLACE
