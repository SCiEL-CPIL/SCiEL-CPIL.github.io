<<<<<<< SEARCH
        # Pruning Phase: remove redundant blocks
        # coverage_count tracks how many selected blocks cover each triple
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_indices:
            for t_idx in block_triples[b_idx]:
                coverage_count[t_idx] += 1
        
        active_blocks = list(selected_indices)
        
        changed = True
        passes = 0
        max_passes = 15
        
        while changed and passes < max_passes:
            if time.time() > deadline:
                break
            changed = False
            passes += 1
            rng.shuffle(active_blocks)
            
            new_active_blocks = []
            for b_idx in active_blocks:
                # Check if any triple in this block has coverage_count == 1
                # If no triple has count == 1, this block is redundant
                block_t_idxs = block_triples[b_idx]
                is_redundant = True
                for t_idx in block_t_idxs:
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for t_idx in block_t_idxs:
                        coverage_count[t_idx] -= 1
                    changed = True
                else:
                    new_active_blocks.append(b_idx)
            
            active_blocks = new_active_blocks
        
        # Convert indices to blocks
        return [list(all_blocks[i]) for i in active_blocks]
=======
        # Pruning Phase: remove redundant blocks
        # coverage_count tracks how many selected blocks cover each triple
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_indices:
            for t_idx in block_triples[b_idx]:
                coverage_count[t_idx] += 1
        
        active_blocks = list(selected_indices)
        
        changed = True
        passes = 0
        max_passes = 15
        
        while changed and passes < max_passes:
            if time.time() > deadline:
                break
            changed = False
            passes += 1
            rng.shuffle(active_blocks)
            
            new_active_blocks = []
            for b_idx in active_blocks:
                # Check if any triple in this block has coverage_count == 1
                # If no triple has count == 1, this block is redundant
                block_t_idxs = block_triples[b_idx]
                is_redundant = True
                for t_idx in block_t_idxs:
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for t_idx in block_t_idxs:
                        coverage_count[t_idx] -= 1
                    changed = True
                else:
                    new_active_blocks.append(b_idx)
            
            active_blocks = new_active_blocks
        
        # Local search: try swapping blocks to reduce total coverage
        if time.time() < deadline - 10:
            active_set = set(active_blocks)
            improved = True
            swap_passes = 0
            max_swap_passes = 5
            
            while improved and swap_passes < max_swap_passes:
                if time.time() > deadline - 5:
                    break
                improved = False
                swap_passes += 1
                
                # Recompute coverage after removals/additions
                coverage_count = np.zeros(n_triples, dtype=np.int32)
                for b_idx in active_blocks:
                    for t_idx in block_triples[b_idx]:
                        coverage_count[t_idx] += 1
                
                # For each active block, find its uniquely covered triples
                # Try to find a non-active block that covers those triples
                # and has lower overlap with the rest
                rng.shuffle(active_blocks)
                
                new_active = []
                for b_idx in active_blocks:
                    # Find triples uniquely covered by this block
                    unique_triples = []
                    for t_idx in block_triples[b_idx]:
                        if coverage_count[t_idx] == 1:
                            unique_triples.append(t_idx)
                    
                    if not unique_triples:
                        # Block is redundant (shouldn't happen after pruning, but safety)
                        for t_idx in block_triples[b_idx]:
                            coverage_count[t_idx] -= 1
                        continue
                    
                    # Try to find a replacement block that covers all unique triples
                    # and has minimal additional coverage
                    best_replace = None
                    best_extra = k_t + 1  # worst case
                    
                    # Sample some non-active blocks
                    n_candidates = min(200, n_blocks - len(active_blocks))
                    if n_candidates > 0:
                        cand_indices = rng.choice(n_blocks, size=n_candidates, replace=False)
                        for cand in cand_indices:
                            if cand in active_set:
                                continue
                            cand_triples = block_triples[cand]
                            # Check if cand covers all unique triples
                            cand_set = set(cand_triples)
                            all_covered = all(t in cand_set for t in unique_triples)
                            if not all_covered:
                                continue
                            # Count how many of cand's triples are already covered
                            extra = 0
                            for t_idx in cand_triples:
                                if coverage_count[t_idx] == 0:
                                    extra += 1
                            if extra < best_extra:
                                best_extra = extra
                                best_replace = cand
                    
                    if best_replace is not None:
                        # Remove old block
                        for t_idx in block_triples[b_idx]:
                            coverage_count[t_idx] -= 1
                        # Add new block
                        for t_idx in block_triples[best_replace]:
                            coverage_count[t_idx] += 1
                        active_set.discard(b_idx)
                        active_set.add(best_replace)
                        new_active.append(best_replace)
                        improved = True
                    else:
                        new_active.append(b_idx)
                
                active_blocks = new_active
        
        # Convert indices to blocks
        return [list(all_blocks[i]) for i in active_blocks]
>>>>>>> REPLACE
