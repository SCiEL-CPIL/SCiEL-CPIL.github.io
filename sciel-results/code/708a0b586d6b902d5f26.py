<<<<<<< SEARCH
    def local_search(blocks_in_sol):
        if not blocks_in_sol:
            return blocks_in_sol
            
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_in_sol:
            for t_idx in block_triples[b_idx]:
                counts[t_idx] += 1
                
        sol_set = set(blocks_in_sol)
        improved = True
        
        while improved:
            improved = False
            
            # 1. Remove redundant
            to_remove = []
            for b_idx in blocks_in_sol:
                is_redundant = True
                for t_idx in block_triples[b_idx]:
                    if counts[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(b_idx)
                    
            if to_remove:
                remove_set = set(to_remove)
                blocks_in_sol = [b for b in blocks_in_sol if b not in remove_set]
                sol_set = set(blocks_in_sol)
                for b_idx in to_remove:
                    for t_idx in block_triples[b_idx]:
                        counts[t_idx] -= 1
                improved = True
                continue
            
            # 2. Swap
            swaps_made = False
            # Shuffle blocks for randomization
            for b_idx in random.sample(blocks_in_sol, len(blocks_in_sol)):
                unique_triples = [t_idx for t_idx in block_triples[b_idx] if counts[t_idx] == 1]
                if not unique_triples:
                    continue
                
                # Find candidates covering all unique_triples
                # Use the first unique triple to filter
                first_t = unique_triples[0]
                candidates = []
                for cand_b_idx in triple_blocks[first_t]:
                    if cand_b_idx in sol_set:
                        continue
                    # Check if covers all unique_triples
                    valid = True
                    for ut in unique_triples:
                        if ut not in block_triples[cand_b_idx]:
                            valid = False
                            break
                    if valid:
                        candidates.append(cand_b_idx)
                
                if not candidates:
                    continue
                
                random.shuffle(candidates)
                # Try more candidates
                for new_b_idx in candidates[:50]: 
                    # Simulate swap
                    # Check if any block becomes redundant
                    # We need to check blocks that might become redundant.
                    # A block C becomes redundant if all its triples have count >= 2 after swap.
                    
                    # To speed up, we can just check all blocks in sol.
                    # But we need to know the NEW counts.
                    # new_counts = counts - block_triples[b_idx] + block_triples[new_b_idx]
                    
                    # Instead of creating a new array, we can check conditionally.
                    # For each block C in sol:
                    #   For each t in C:
                    #      if t in B and t not in B': new_count = count[t] - 1
                    #      if t in B' and t not in B: new_count = count[t] + 1
                    #      else: new_count = count[t]
                    #      if new_count == 1: C is not redundant.
                    
                    # This is O(N_sol * 35 * 35) ? No.
                    # We can precompute sets of triples for B and B'.
                    b_set = set(block_triples[b_idx])
                    nb_set = set(block_triples[new_b_idx])
                    
                    found_redundant = False
                    redundant_block = -1
                    
                    for c_idx in blocks_in_sol:
                        if c_idx == b_idx:
                            continue # b_idx is being removed
                        is_redundant = True
                        for t_idx in block_triples[c_idx]:
                            # Determine new count for t_idx
                            if t_idx in b_set and t_idx not in nb_set:
                                new_c = counts[t_idx] - 1
                            elif t_idx in nb_set and t_idx not in b_set:
                                new_c = counts[t_idx] + 1
                            else:
                                new_c = counts[t_idx]
                                
                            if new_c == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            found_redundant = True
                            redundant_block = c_idx
                            break
                    
                    if found_redundant:
                        # Commit swap and removal
                        sol_set.remove(b_idx)
                        sol_set.add(new_b_idx)
                        blocks_in_sol.remove(b_idx)
                        blocks_in_sol.append(new_b_idx)
                        
                        for t_idx in block_triples[b_idx]:
                            counts[t_idx] -= 1
                        for t_idx in block_triples[new_b_idx]:
                            counts[t_idx] += 1
                        
                        # Remove redundant block
                        sol_set.remove(redundant_block)
                        blocks_in_sol.remove(redundant_block)
                        for t_idx in block_triples[redundant_block]:
                            counts[t_idx] -= 1
                            
                        improved = True
                        swaps_made = True
                        break # Restart main loop
                    else:
                        # No swap, so no need to revert counts (we didn't change them)
                        pass
                if swaps_made:
                    break
            
        return blocks_in_sol
=======
    def local_search(blocks_in_sol):
        if not blocks_in_sol:
            return blocks_in_sol
            
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_in_sol:
            for t_idx in block_triples[b_idx]:
                counts[t_idx] += 1
                
        sol_set = set(blocks_in_sol)
        improved = True
        max_iter = 50  # Limit iterations to avoid timeout
        
        while improved and max_iter > 0:
            max_iter -= 1
            improved = False
            
            # 1. Remove redundant
            to_remove = []
            for b_idx in blocks_in_sol:
                is_redundant = True
                for t_idx in block_triples[b_idx]:
                    if counts[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(b_idx)
                    
            if to_remove:
                remove_set = set(to_remove)
                blocks_in_sol = [b for b in blocks_in_sol if b not in remove_set]
                sol_set = set(blocks_in_sol)
                for b_idx in to_remove:
                    for t_idx in block_triples[b_idx]:
                        counts[t_idx] -= 1
                improved = True
                continue
            
            # 2. Swap with potential removal
            swaps_made = False
            # Shuffle blocks for randomization
            for b_idx in random.sample(blocks_in_sol, len(blocks_in_sol)):
                unique_triples = [t_idx for t_idx in block_triples[b_idx] if counts[t_idx] == 1]
                if not unique_triples:
                    continue
                
                # Find candidates covering all unique_triples
                first_t = unique_triples[0]
                candidates = []
                for cand_b_idx in triple_blocks[first_t]:
                    if cand_b_idx in sol_set:
                        continue
                    valid = True
                    for ut in unique_triples:
                        if ut not in block_triple_set[cand_b_idx]:
                            valid = False
                            break
                    if valid:
                        candidates.append(cand_b_idx)
                
                if not candidates:
                    continue
                
                random.shuffle(candidates)
                for new_b_idx in candidates[:30]: 
                    b_set = block_triple_set[b_idx]
                    nb_set = block_triple_set[new_b_idx]
                    
                    # Check if any block becomes redundant after swap
                    found_redundant = False
                    redundant_block = -1
                    
                    for c_idx in blocks_in_sol:
                        if c_idx == b_idx:
                            continue
                        is_redundant = True
                        for t_idx in block_triples[c_idx]:
                            if t_idx in b_set and t_idx not in nb_set:
                                new_c = counts[t_idx] - 1
                            elif t_idx in nb_set and t_idx not in b_set:
                                new_c = counts[t_idx] + 1
                            else:
                                new_c = counts[t_idx]
                                
                            if new_c == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            found_redundant = True
                            redundant_block = c_idx
                            break
                    
                    if found_redundant:
                        # Commit swap and removal
                        sol_set.remove(b_idx)
                        sol_set.add(new_b_idx)
                        blocks_in_sol.remove(b_idx)
                        blocks_in_sol.append(new_b_idx)
                        
                        for t_idx in block_triples[b_idx]:
                            counts[t_idx] -= 1
                        for t_idx in block_triples[new_b_idx]:
                            counts[t_idx] += 1
                        
                        sol_set.remove(redundant_block)
                        blocks_in_sol.remove(redundant_block)
                        for t_idx in block_triples[redundant_block]:
                            counts[t_idx] -= 1
                            
                        improved = True
                        swaps_made = True
                        break
                if swaps_made:
                    break
            
            # 3. Try adding a block that enables removal of two blocks
            if not improved and len(blocks_in_sol) > 2:
                add_made = False
                for _ in range(20):  # Try a few random additions
                    # Pick two blocks to potentially remove
                    if len(blocks_in_sol) < 2:
                        break
                    b1, b2 = random.sample(blocks_in_sol, 2)
                    
                    # Find triples only covered by b1 or b2
                    b1_set = block_triple_set[b1]
                    b2_set = block_triple_set[b2]
                    
                    unique_to_both = []
                    for t_idx in b1_set:
                        if counts[t_idx] == 1:
                            unique_to_both.append(t_idx)
                    for t_idx in b2_set:
                        if counts[t_idx] == 1 and t_idx not in b1_set:
                            unique_to_both.append(t_idx)
                    
                    if not unique_to_both:
                        continue
                    
                    # Find a block that covers all unique_to_both
                    first_t = unique_to_both[0]
                    found_new_block = False
                    for cand_b_idx in triple_blocks[first_t]:
                        if cand_b_idx in sol_set:
                            continue
                        valid = True
                        for ut in unique_to_both:
                            if ut not in block_triple_set[cand_b_idx]:
                                valid = False
                                break
                        if valid:
                            # Check if removing b1 and b2 is valid
                            # All triples in b1 and b2 must be covered by new block or other blocks
                            can_remove = True
                            for t_idx in b1_set | b2_set:
                                if counts[t_idx] <= 2 and t_idx not in block_triple_set[cand_b_idx]:
                                    can_remove = False
                                    break
                                elif counts[t_idx] == 1 and t_idx not in block_triple_set[cand_b_idx]:
                                    can_remove = False
                                    break
                            
                            if can_remove:
                                # Commit
                                sol_set.remove(b1)
                                sol_set.remove(b2)
                                sol_set.add(cand_b_idx)
                                blocks_in_sol.remove(b1)
                                blocks_in_sol.remove(b2)
                                blocks_in_sol.append(cand_b_idx)
                                
                                for t_idx in block_triples[b1]:
                                    counts[t_idx] -= 1
                                for t_idx in block_triples[b2]:
                                    counts[t_idx] -= 1
                                for t_idx in block_triples[cand_b_idx]:
                                    counts[t_idx] += 1
                                    
                                improved = True
                                add_made = True
                                break
                        if found_new_block:
                            break
                    if add_made:
                        break
                if add_made:
                    continue
            
        return blocks_in_sol
>>>>>>> REPLACE

