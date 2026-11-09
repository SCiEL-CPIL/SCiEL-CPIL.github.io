<<<<<<< SEARCH
    # Local Search: Removal + Swap
    def refine(indices_selected):
        if not indices_selected:
            return indices_selected
            
        counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices_selected:
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            counts += cover_row
            
        # Removal Pass
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_indices = []
            for idx in indices_selected:
                chunk_idx = idx // chunk_size
                local_idx = idx % chunk_size
                cover_row = cover_list[chunk_idx][local_idx]
                
                critical_mask = (cover_row == 1) & (counts == 1)
                if not np.any(critical_mask):
                    counts -= cover_row
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices_selected = new_indices
            
        # Swap Pass: Try to replace blocks with ones that enable further removals
        if time.time() < deadline - 5:
            for _ in range(3):  # 3 swap passes
                if time.time() > deadline - 5:
                    break
                improved = False
                for i_pos in range(len(indices_selected)):
                    if time.time() > deadline - 5:
                        break
                    idx_i = indices_selected[i_pos]
                    chunk_i = idx_i // chunk_size
                    local_i = idx_i % chunk_size
                    cover_i = cover_list[chunk_i][local_i]
                    
                    critical_mask = (cover_i == 1) & (counts == 1)
                    if not np.any(critical_mask):
                        continue
                        
                    # Find candidate blocks that cover all critical subsets
                    # and maximize redundancy coverage
                    best_swap = -1
                    best_score = -1
                    
                    for chunk_j, cover_chunk in enumerate(cover_list):
                        if time.time() > deadline - 5:
                            break
                        # Check which blocks in this chunk cover all critical subsets
                        # (cover_chunk & critical_mask[None, :]).all(axis=1)
                        covers_critical = (cover_chunk & critical_mask[None, :]).all(axis=1)
                        
                        if not np.any(covers_critical):
                            continue
                            
                        candidate_indices = np.where(covers_critical)[0]
                        
                        for local_j in candidate_indices:
                            global_j = chunk_j * chunk_size + local_j
                            if global_j == idx_i:
                                continue
                            # Check if already in selected
                            if global_j in indices_selected:
                                continue
                                
                            cover_j = cover_chunk[local_j]
                            
                            # Score: number of subsets with count >= 2 that j covers
                            # (these are "redundant" and swapping might allow removal)
                            redundancy = np.sum((counts >= 2) & (cover_j == 1))
                            
                            if redundancy > best_score:
                                best_score = redundancy
                                best_swap = global_j
                    
                    if best_swap != -1:
                        # Perform the swap
                        chunk_swap = best_swap // chunk_size
                        local_swap = best_swap % chunk_size
                        cover_swap = cover_list[chunk_swap][local_swap]
                        
                        counts = counts - cover_i + cover_swap
                        
                        # Update indices_selected
                        indices_selected[i_pos] = best_swap
                        improved = True
                        
                        # Try removal again after swap
                        changed = True
                        while changed and time.time() < deadline - 5:
                            changed = False
                            new_indices = []
                            for idx in indices_selected:
                                chunk_idx = idx // chunk_size
                                local_idx = idx % chunk_size
                                cover_row = cover_list[chunk_idx][local_idx]
                                
                                critical_mask2 = (cover_row == 1) & (counts == 1)
                                if not np.any(critical_mask2):
                                    counts -= cover_row
                                    changed = True
                                else:
                                    new_indices.append(idx)
                            indices_selected = new_indices
                            
                            if len(indices_selected) < i_pos + 1:
                                break
                    
                if not improved:
                    break
        
        return indices_selected
=======
    # Local Search: Removal + Swap
    def refine(indices_selected):
        if not indices_selected:
            return indices_selected
            
        counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices_selected:
            counts += coverage_matrix[idx]
            
        # Removal Pass
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_indices = []
            for idx in indices_selected:
                row = coverage_matrix[idx]
                
                critical_mask = (row == 1) & (counts == 1)
                if not np.any(critical_mask):
                    counts -= row
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices_selected = new_indices
            
        # Swap Pass: Try to replace blocks with ones that enable further removals
        if time.time() < deadline - 5:
            for _ in range(2):  # 2 swap passes
                if time.time() > deadline - 5:
                    break
                improved = False
                for i_pos in range(len(indices_selected)):
                    if time.time() > deadline - 5:
                        break
                    idx_i = indices_selected[i_pos]
                    row_i = coverage_matrix[idx_i]
                    
                    critical_mask = (row_i == 1) & (counts == 1)
                    if not np.any(critical_mask):
                        continue
                        
                    # Find candidate blocks that cover all critical subsets
                    # Use the first critical subset to filter candidates
                    crit_indices = np.where(critical_mask)[0]
                    first_crit = crit_indices[0]
                    
                    # Get all blocks that cover first_crit
                    col_first = coverage_matrix[:, first_crit]
                    candidates = np.where(col_first == 1)[0]
                    
                    best_swap = -1
                    best_score = -1
                    
                    for cand_idx in candidates:
                        if cand_idx in indices_selected:
                            continue
                        row_cand = coverage_matrix[cand_idx]
                        
                        # Check if cand covers ALL critical subsets
                        if not np.all(row_cand[crit_indices] == 1):
                            continue
                        
                        # Score: maximize redundancy
                        redundancy = np.sum((counts >= 2) & (row_cand == 1))
                        
                        if redundancy > best_score:
                            best_score = redundancy
                            best_swap = cand_idx
                            
                    if best_swap != -1:
                        row_swap = coverage_matrix[best_swap]
                        counts = counts - row_i + row_swap
                        indices_selected[i_pos] = best_swap
                        improved = True
                        
                        # Re-run removal
                        changed = True
                        while changed and time.time() < deadline - 5:
                            changed = False
                            new_indices = []
                            for idx in indices_selected:
                                row = coverage_matrix[idx]
                                critical = (row == 1) & (counts == 1)
                                if not np.any(critical):
                                    counts -= row
                                    changed = True
                                else:
                                    new_indices.append(idx)
                            indices_selected = new_indices
                            
                if not improved:
                    break
        
        return indices_selected
>>>>>>> REPLACE

Finally, update the main loop validation and return.

