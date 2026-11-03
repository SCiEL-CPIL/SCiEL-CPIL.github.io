<<<<<<< SEARCH
    def local_search(selected):
        """Aggressive local search: removal + 1-swap + 2-swap using set intersections."""
        if not selected:
            return selected
        
        selected = list(selected)
        selected_set = set(selected)
        
        # Initialize counts
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            counts[block_t_indices[b_idx]] += 1
        
        # Phase 1: Iterative removal of redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected:
                ts = block_t_indices[b_idx]
                if np.all(counts[ts] > 1):
                    counts[ts] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Phase 2: 1-swap local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            # Recompute counts
            counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in new_selected:
                counts[block_t_indices[b_idx]] += 1
            
            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline:
                    break
                
                ts = block_t_indices[b_idx]
                # Find t-subsets uniquely covered by b_idx
                unique_mask = counts[ts] == 1
                if not unique_mask.any():
                    # Redundant
                    counts[ts] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break
                
                unique_indices = ts[unique_mask]
                
                # Find candidates that cover all unique t-subsets
                if len(unique_indices) == 0:
                    continue
                    
                candidates = t_to_blocks_sets[unique_indices[0]]
                for t_idx in unique_indices[1:]:
                    candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                
                candidates.difference_update(new_set)
                
                if candidates:
                    # Pick candidate with best additional coverage
                    best_cand = None
                    best_score = 0
                    cand_list = list(candidates)
                    # Sample if too many
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    for cand in cand_list:
                        # Score: additional t-subsets covered
                        diff = block_masks[cand] & ~block_masks[b_idx]
                        score = diff.bit_count()
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    
                    if best_cand is not None and best_score > 0:
                        # Perform swap
                        counts[block_t_indices[b_idx]] -= 1
                        counts[block_t_indices[best_cand]] += 1
                        new_selected.remove(b_idx)
                        new_selected.append(best_cand)
                        new_set.discard(b_idx)
                        new_set.add(best_cand)
                        improved = True
                        break
            
            selected = new_selected
        
        # Phase 3: 2-swap local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in new_selected:
                counts[block_t_indices[b_idx]] += 1
            
            n_sel = len(new_selected)
            if n_sel < 2:
                break
            
            for i in range(n_sel):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_sel):
                    b1 = new_selected[i]
                    b2 = new_selected[j]
                    
                    # Find t-subsets uniquely covered by b1 or b2
                    ts1 = block_t_indices[b1]
                    ts2 = block_t_indices[b2]
                    
                    # t-subsets in b1 not in b2 with count==1
                    set1 = set(ts1)
                    set2 = set(ts2)
                    only1 = set1 - set2
                    only2 = set2 - set1
                    
                    unique_to_pair = []
                    for idx in only1:
                        if counts[idx] == 1:
                            unique_to_pair.append(idx)
                    for idx in only2:
                        if counts[idx] == 1:
                            unique_to_pair.append(idx)
                    
                    if not unique_to_pair:
                        # Both redundant
                        counts[ts1] -= 1
                        counts[ts2] -= 1
                        new_selected.remove(b1)
                        new_selected.remove(b2)
                        new_set.discard(b1)
                        new_set.discard(b2)
                        improved = True
                        break
                    
                    # Find 2 blocks that cover all unique_to_pair
                    # Use intersection approach
                    candidates_from_first = t_to_blocks_sets[unique_to_pair[0]]
                    
                    best_pair = None
                    best_score = 0
                    
                    cand_list = list(candidates_from_first)
                    if len(cand_list) > 30:
                        cand_list = random.sample(cand_list, 30)
                    
                    for c1 in cand_list:
                        if c1 in new_set or c1 == b1 or c1 == b2:
                            continue
                        
                        # Check what c1 covers of unique_to_pair
                        c1_mask = block_masks[c1]
                        remaining = [ui for ui in unique_to_pair if not (c1_mask >> ui) & 1]
                        
                        if not remaining:
                            # c1 covers all, find good c2
                            best_c2 = None
                            best_c2_score = 0
                            c2_candidates = t_to_blocks_lists[unique_to_pair[0]]
                            for c2 in c2_candidates:
                                if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                    continue
                                diff = (block_masks[c2] & ~block_masks[b1] & ~block_masks[b2])
                                score = diff.bit_count()
                                if score > best_c2_score:
                                    best_c2_score = score
                                    best_c2 = c2
                            if best_c2 is not None and best_c2_score > 0:
                                if best_c2_score > best_score:
                                    best_score = best_c2_score
                                    best_pair = (c1, best_c2)
                        else:
                            # Need c2 to cover remaining
                            c2_candidates = t_to_blocks_lists[remaining[0]]
                            for c2 in c2_candidates:
                                if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                    continue
                                # Check if c2 covers all remaining
                                c2_mask = block_masks[c2]
                                covers = all((c2_mask >> ui) & 1 for ui in remaining)
                                if not covers:
                                    continue
                                combined_new = (block_masks[c1] | block_masks[c2]) & ~block_masks[b1] & ~block_masks[b2]
                                score = combined_new.bit_count()
                                if score > best_score:
                                    best_score = score
                                    best_pair = (c1, c2)
                    
                    if best_pair and best_score > 0:
                        c1, c2 = best_pair
                        counts[ts1] -= 1
                        counts[ts2] -= 1
                        counts[block_t_indices[c1]] += 1
                        counts[block_t_indices[c2]] += 1
                        new_selected.remove(b1)
                        new_selected.remove(b2)
                        new_selected.append(c1)
                        new_selected.append(c2)
                        new_set.discard(b1)
                        new_set.discard(b2)
                        new_set.add(c1)
                        new_set.add(c2)
                        improved = True
                        break
                
                if improved:
                    break
            
            selected = new_selected
        
        return selected
=======
    def local_search(selected):
        """Aggressive local search: removal + 1-swap + 2-swap using set intersections."""
        if not selected:
            return selected
        
        selected = list(selected)
        selected_set = set(selected)
        
        # Initialize counts
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            counts[block_t_indices[b_idx]] += 1
        
        # Phase 1: Iterative removal of redundant blocks
        changed = True
        while changed and time.time() < deadline - 10: # Leave time for later phases
            changed = False
            new_selected = []
            for b_idx in selected:
                ts = block_t_indices[b_idx]
                if np.all(counts[ts] > 1):
                    counts[ts] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Phase 2: 1-swap local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            # Recompute counts
            counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in new_selected:
                counts[block_t_indices[b_idx]] += 1
            
            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline - 5:
                    break
                
                ts = block_t_indices[b_idx]
                # Find t-subsets uniquely covered by b_idx
                unique_mask = counts[ts] == 1
                if not unique_mask.any():
                    # Redundant
                    counts[ts] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break
                
                unique_indices = ts[unique_mask]
                
                # Find candidates that cover all unique t-subsets
                if len(unique_indices) == 0:
                    continue
                    
                candidates = t_to_blocks_sets[unique_indices[0]]
                for t_idx in unique_indices[1:]:
                    candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                
                candidates.difference_update(new_set)
                
                if candidates:
                    # Pick candidate with best additional coverage
                    best_cand = None
                    best_score = 0
                    cand_list = list(candidates)
                    # Sample if too many
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    for cand in cand_list:
                        # Score: additional t-subsets covered
                        diff = block_masks[cand] & ~block_masks[b_idx]
                        score = diff.bit_count()
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    
                    if best_cand is not None and best_score > 0:
                        # Perform swap
                        counts[block_t_indices[b_idx]] -= 1
                        counts[block_t_indices[best_cand]] += 1
                        new_selected.remove(b_idx)
                        new_selected.append(best_cand)
                        new_set.discard(b_idx)
                        new_set.add(best_cand)
                        improved = True
                        break
            
            selected = new_selected
        
        # Phase 3: 2-swap local search (Only if time permits)
        if time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                new_selected = list(selected)
                new_set = set(new_selected)
                
                counts = np.zeros(n_t, dtype=np.int32)
                for b_idx in new_selected:
                    counts[block_t_indices[b_idx]] += 1
                
                n_sel = len(new_selected)
                if n_sel < 2:
                    break
                
                for i in range(n_sel):
                    if time.time() > deadline - 1:
                        break
                    for j in range(i + 1, n_sel):
                        b1 = new_selected[i]
                        b2 = new_selected[j]
                        
                        # Find t-subsets uniquely covered by b1 or b2
                        ts1 = block_t_indices[b1]
                        ts2 = block_t_indices[b2]
                        
                        # t-subsets in b1 not in b2 with count==1
                        set1 = set(ts1)
                        set2 = set(ts2)
                        only1 = set1 - set2
                        only2 = set2 - set1
                        
                        unique_to_pair = []
                        for idx in only1:
                            if counts[idx] == 1:
                                unique_to_pair.append(idx)
                        for idx in only2:
                            if counts[idx] == 1:
                                unique_to_pair.append(idx)
                        
                        if not unique_to_pair:
                            # Both redundant
                            counts[ts1] -= 1
                            counts[ts2] -= 1
                            new_selected.remove(b1)
                            new_selected.remove(b2)
                            new_set.discard(b1)
                            new_set.discard(b2)
                            improved = True
                            break
                        
                        # Find 2 blocks that cover all unique_to_pair
                        # Use intersection approach
                        candidates_from_first = t_to_blocks_sets[unique_to_pair[0]]
                        
                        best_pair = None
                        best_score = 0
                        
                        cand_list = list(candidates_from_first)
                        if len(cand_list) > 30:
                            cand_list = random.sample(cand_list, 30)
                        
                        for c1 in cand_list:
                            if c1 in new_set or c1 == b1 or c1 == b2:
                                continue
                            
                            # Check what c1 covers of unique_to_pair
                            c1_mask = block_masks[c1]
                            remaining = [ui for ui in unique_to_pair if not (c1_mask >> ui) & 1]
                            
                            if not remaining:
                                # c1 covers all, find good c2
                                best_c2 = None
                                best_c2_score = 0
                                c2_candidates = t_to_blocks_lists[unique_to_pair[0]]
                                for c2 in c2_candidates:
                                    if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                        continue
                                    diff = (block_masks[c2] & ~block_masks[b1] & ~block_masks[b2])
                                    score = diff.bit_count()
                                    if score > best_c2_score:
                                        best_c2_score = score
                                        best_c2 = c2
                                if best_c2 is not None and best_c2_score > 0:
                                    if best_c2_score > best_score:
                                        best_score = best_c2_score
                                        best_pair = (c1, best_c2)
                            else:
                                # Need c2 to cover remaining
                                c2_candidates = t_to_blocks_lists[remaining[0]]
                                for c2 in c2_candidates:
                                    if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                        continue
                                    # Check if c2 covers all remaining
                                    c2_mask = block_masks[c2]
                                    covers = all((c2_mask >> ui) & 1 for ui in remaining)
                                    if not covers:
                                        continue
                                    combined_new = (block_masks[c1] | block_masks[c2]) & ~block_masks[b1] & ~block_masks[b2]
                                    score = combined_new.bit_count()
                                    if score > best_score:
                                        best_score = score
                                        best_pair = (c1, c2)
                        
                        if best_pair and best_score > 0:
                            c1, c2 = best_pair
                            counts[ts1] -= 1
                            counts[ts2] -= 1
                            counts[block_t_indices[c1]] += 1
                            counts[block_t_indices[c2]] += 1
                            new_selected.remove(b1)
                            new_selected.remove(b2)
                            new_selected.append(c1)
                            new_selected.append(c2)
                            new_set.discard(b1)
                            new_set.discard(b2)
                            new_set.add(c1)
                            new_set.add(c2)
                            improved = True
                            break
                    
                    if improved:
                        break
                
                selected = new_selected
        
        return selected

    def ensure_validity(selected_indices):
        """
        Checks if selected_indices covers all t-subsets.
        If not, greedily adds blocks to cover the missing ones.
        Returns the updated list of block indices.
        """
        if not selected_indices:
            return []
            
        # Count coverage
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_indices:
            counts[block_t_indices[b_idx]] += 1
            
        uncovered_mask = counts == 0
        
        if not uncovered_mask.any():
            return selected_indices
            
        # Greedily cover uncovered
        selected_list = list(selected_indices)
        uncovered_set = set(np.where(uncovered_mask)[0].tolist())
        
        # While there are uncovered t-subsets
        while uncovered_set:
            if time.time() > deadline - 0.5: # Very tight margin to avoid timeout
                break
                
            # Find block that covers most uncovered
            target_t = next(iter(uncovered_set))
            candidates = t_to_blocks_lists[target_t]
            
            best_block = -1
            best_count = 0
            
            for cand in candidates:
                # Count how many uncovered sets this candidate covers
                cov = 0
                for t_idx in block_t_indices[cand]:
                    if t_idx in uncovered_set:
                        cov += 1
                if cov > best_count:
                    best_count = cov
                    best_block = cand
                    
            if best_block == -1:
                break
                
            selected_list.append(best_block)
            # Update uncovered_set
            for t_idx in block_t_indices[best_block]:
                if t_idx in uncovered_set:
                    uncovered_set.remove(t_idx)
                    
        return selected_list
>>>>>>> REPLACE

Now update the main logic to use `ensure_validity`.

