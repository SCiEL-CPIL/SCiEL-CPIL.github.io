    def local_search(blocks):
        selected_blocks = list(blocks)
        selected_set = set(selected_blocks)
        cover_count = [0] * num_t_subsets
        
        for b_idx in selected_blocks:
            for idx in block_t_indices[b_idx]:
                cover_count[idx] += 1
                
        improved = True
        # Use a local deadline to avoid eating all time
        # We don't have a specific time per call, but we can check global deadline
        local_deadline = deadline - 2.0
        
        while improved and time.time() < local_deadline:
            improved = False
            
            # Pass 1: Reduction
            i = 0
            while i < len(selected_blocks):
                b_idx = selected_blocks[i]
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    selected_blocks.pop(i)
                    selected_set.remove(b_idx)
                    for idx in block_t_indices[b_idx]:
                        cover_count[idx] -= 1
                    improved = True
                else:
                    i += 1
                    
            if time.time() >= local_deadline:
                break
                
            # Pass 2: 1-Swap
            swapped = False
            for i in range(len(selected_blocks)):
                if time.time() >= local_deadline:
                    break
                    
                b_idx = selected_blocks[i]
                
                # Find critical t-subsets
                critical_ts = []
                for idx in block_t_indices[b_idx]:
                    if cover_count[idx] == 1:
                        critical_ts.append(idx)
                        
                if not critical_ts:
                    # Redundant
                    selected_blocks.pop(i)
                    selected_set.remove(b_idx)
                    for idx in block_t_indices[b_idx]:
                        cover_count[idx] -= 1
                    improved = True
                    swapped = True
                    break
                    
                # Candidate blocks must cover all critical_ts
                # Intersect t_to_blocks for all critical_ts
                # Start with the smallest list
                critical_ts.sort(key=lambda x: len(t_to_blocks[x]))
                candidates = t_to_blocks[critical_ts[0]]
                
                valid_candidate = -1
                for j in candidates:
                    if j in selected_set:
                        continue
                    # Check if j covers all other critical_ts
                    covers_all = True
                    # Optimization: Check if j is in t_to_blocks for other criticals
                    # Since we don't have a fast "is in" for lists, we can use a set for critical_ts?
                    # Or just check block_t_indices[j]
                    j_indices = set(block_t_indices[j])
                    for c in critical_ts[1:]:
                        if c not in j_indices:
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidate = j
                        break
                        
                if valid_candidate != -1:
                    # Swap
                    selected_blocks[i] = valid_candidate
                    selected_set.remove(b_idx)
                    selected_set.add(valid_candidate)
                    
                    for idx in block_t_indices[b_idx]:
                        cover_count[idx] -= 1
                    for idx in block_t_indices[valid_candidate]:
                        cover_count[idx] += 1
                            
                    improved = True
                    swapped = True
                    break # Restart outer loop
                else:
                    # No swap found for this block, move to next
                    pass
                    
            if not swapped:
                # No swaps found in this pass
                break
                
        return selected_blocks
