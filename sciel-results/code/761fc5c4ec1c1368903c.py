<<<<<<< SEARCH
    # Helper: Local Search with efficient 2-1 exchanges
    def local_search(selected):
        if not selected:
            return selected
        
        current = list(selected)
        
        # Build cover_count
        cover_count = [0] * n_t_subs
        for bi in current:
            for idx in block_t_indices[bi]:
                cover_count[idx] += 1
        
        # Set of selected blocks for fast lookup
        selected_set = set(current)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Phase 1: Remove redundant blocks
            new_current = []
            for bi in current:
                is_redundant = True
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_t_indices[bi]:
                        cover_count[idx] -= 1
                    selected_set.remove(bi)
                    improved = True
                else:
                    new_current.append(bi)
            current = new_current
            
            if time.time() > deadline - 10:
                break
                
            # Phase 2: 2-1 Exchange (remove 2 blocks, add 1 if needed)
            n = len(current)
            if n < 2:
                continue
                
            # Build a list of (block_idx, its_t_indices) for quick access
            # We'll iterate over pairs in random order
            pair_list = []
            for i in range(n):
                for j in range(i+1, n):
                    pair_list.append((i, j))
            random.shuffle(pair_list)
            
            found_improvement = False
            for i, j in pair_list:
                if time.time() > deadline - 10:
                    break
                if i >= len(current) or j >= len(current):
                    continue
                
                bi = current[i]
                bj = current[j]
                
                # Compute the union of t-subsets covered by bi and bj
                # mask_bi | mask_bj gives us all t-subsets they cover together
                # The t-subsets that become uncovered are those where cover_count == 1 (for bi) 
                # or cover_count == 1 (for bj) or cover_count == 2 and both cover it
                # More precisely: a t-subset idx is uncovered after removing bi,bj iff
                # cover_count[idx] - (1 if idx in bi else 0) - (1 if idx in bj else 0) == 0
                
                # Efficient approach: 
                # Let union_mask = block_masks[bi] | block_masks[bj]
                # For each idx in union_mask, check if it's still covered
                # But we need to be careful about overlaps.
                
                # Alternative: decrement counts, check, then restore
                # This is O(|bi| + |bj|) which is ~35+35 = 70 per pair
                # With ~10000 pairs, that's 700k operations, which is fine.
                
                for idx in block_t_indices[bi]:
                    cover_count[idx] -= 1
                for idx in block_t_indices[bj]:
                    cover_count[idx] -= 1
                
                # Check if any t-subset became uncovered
                uncovered_indices = []
                # Check indices in bi
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 0:
                        uncovered_indices.append(idx)
                # Check indices in bj (avoid duplicates)
                for idx in block_t_indices[bj]:
                    if cover_count[idx] == 0 and idx not in uncovered_indices:
                        uncovered_indices.append(idx)
                
                if not uncovered_indices:
                    # Can remove both blocks
                    current.remove(bj)
                    current.remove(bi)
                    selected_set.remove(bj)
                    selected_set.remove(bi)
                    improved = True
                    found_improvement = True
                    break
                else:
                    # Try to find a block bk that covers all uncovered_indices
                    # and is not bi or bj
                    # Sort uncovered by number of covering blocks (rarity first)
                    uncovered_indices.sort(key=lambda x: len(t_to_blocks[x]))
                    
                    candidates = None
                    for idx in uncovered_indices:
                        blocks_covering = t_to_blocks[idx]
                        if candidates is None:
                            candidates = set(blocks_covering)
                        else:
                            candidates.intersection_update(blocks_covering)
                        if not candidates:
                            break
                    
                    if candidates:
                        # Filter: bk must not be in selected_set (to avoid adding duplicate)
                        # and must not be bi or bj
                        valid = [bk for bk in candidates if bk not in selected_set and bk != bi and bk != bj]
                        
                        if valid:
                            bk = random.choice(valid)
                            
                            # Remove bi, bj
                            current.remove(bj)
                            current.remove(bi)
                            selected_set.remove(bj)
                            selected_set.remove(bi)
                            
                            # Add bk
                            current.append(bk)
                            selected_set.add(bk)
                            
                            # Update cover_count: we already decremented bi, bj
                            # Now increment for bk
                            for idx in block_t_indices[bk]:
                                cover_count[idx] += 1
                            
                            improved = True
                            found_improvement = True
                            break
                    else:
                        # No valid replacement, restore counts
                        for idx in block_t_indices[bi]:
                            cover_count[idx] += 1
                        for idx in block_t_indices[bj]:
                            cover_count[idx] += 1
                
            if found_improvement:
                continue
        
        return current
=======
    # Helper: Local Search with efficient 1-0, 2-1, and 3-2 exchanges
    def local_search(selected):
        if not selected:
            return selected
        
        current = list(selected)
        
        # Build cover_count
        cover_count = [0] * n_t_subs
        for bi in current:
            for idx in block_t_indices[bi]:
                cover_count[idx] += 1
        
        # Set of selected blocks for fast lookup
        selected_set = set(current)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Phase 1: Remove redundant blocks (1-0 exchange)
            new_current = []
            for bi in current:
                is_redundant = True
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_t_indices[bi]:
                        cover_count[idx] -= 1
                    selected_set.remove(bi)
                    improved = True
                else:
                    new_current.append(bi)
            current = new_current
            
            if time.time() > deadline - 10:
                break
                
            # Phase 2: 2-1 Exchange (remove 2 blocks, add 1 if needed)
            n = len(current)
            if n >= 2:
                pair_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        pair_list.append((i, j))
                random.shuffle(pair_list)
                
                found_improvement = False
                for i, j in pair_list:
                    if time.time() > deadline - 10:
                        break
                    if i >= len(current) or j >= len(current):
                        continue
                    
                    bi = current[i]
                    bj = current[j]
                    
                    for idx in block_t_indices[bi]:
                        cover_count[idx] -= 1
                    for idx in block_t_indices[bj]:
                        cover_count[idx] -= 1
                    
                    uncovered_indices = []
                    for idx in block_t_indices[bi]:
                        if cover_count[idx] == 0:
                            uncovered_indices.append(idx)
                    for idx in block_t_indices[bj]:
                        if cover_count[idx] == 0 and idx not in uncovered_indices:
                            uncovered_indices.append(idx)
                    
                    if not uncovered_indices:
                        current.remove(bj)
                        current.remove(bi)
                        selected_set.remove(bj)
                        selected_set.remove(bi)
                        improved = True
                        found_improvement = True
                        break
                    else:
                        uncovered_indices.sort(key=lambda x: len(t_to_blocks[x]))
                        
                        candidates = None
                        for idx in uncovered_indices:
                            blocks_covering = t_to_blocks[idx]
                            if candidates is None:
                                candidates = set(blocks_covering)
                            else:
                                candidates.intersection_update(blocks_covering)
                            if not candidates:
                                break
                        
                        if candidates:
                            valid = [bk for bk in candidates if bk not in selected_set and bk != bi and bk != bj]
                            
                            if valid:
                                bk = random.choice(valid)
                                current.remove(bj)
                                current.remove(bi)
                                selected_set.remove(bj)
                                selected_set.remove(bi)
                                current.append(bk)
                                selected_set.add(bk)
                                for idx in block_t_indices[bk]:
                                    cover_count[idx] += 1
                                improved = True
                                found_improvement = True
                                break
                        else:
                            for idx in block_t_indices[bi]:
                                cover_count[idx] += 1
                            for idx in block_t_indices[bj]:
                                cover_count[idx] += 1
                    
            if found_improvement:
                continue
            
            # Phase 3: 3-2 Exchange (remove 3 blocks, add 2 if needed)
            n = len(current)
            if n >= 3:
                triple_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        for l in range(j+1, n):
                            triple_list.append((i, j, l))
                random.shuffle(triple_list)
                
                # Limit number of triples to check to stay within time budget
                max_triples_to_check = min(len(triple_list), 500)
                
                found_improvement = False
                for i, j, l in triple_list[:max_triples_to_check]:
                    if time.time() > deadline - 10:
                        break
                    if i >= len(current) or j >= len(current) or l >= len(current):
                        continue
                    
                    bi = current[i]
                    bj = current[j]
                    bl = current[l]
                    
                    # Decrement counts for all three blocks
                    for idx in block_t_indices[bi]:
                        cover_count[idx] -= 1
                    for idx in block_t_indices[bj]:
                        cover_count[idx] -= 1
                    for idx in block_t_indices[bl]:
                        cover_count[idx] -= 1
                    
                    # Find uncovered t-subsets
                    uncovered_indices = []
                    seen_uncovered = set()
                    for idx in block_t_indices[bi]:
                        if cover_count[idx] == 0:
                            uncovered_indices.append(idx)
                            seen_uncovered.add(idx)
                    for idx in block_t_indices[bj]:
                        if cover_count[idx] == 0 and idx not in seen_uncovered:
                            uncovered_indices.append(idx)
                            seen_uncovered.add(idx)
                    for idx in block_t_indices[bl]:
                        if cover_count[idx] == 0 and idx not in seen_uncovered:
                            uncovered_indices.append(idx)
                            seen_uncovered.add(idx)
                    
                    if not uncovered_indices:
                        # Can remove all three blocks
                        current.remove(bl)
                        current.remove(bj)
                        current.remove(bi)
                        selected_set.remove(bl)
                        selected_set.remove(bj)
                        selected_set.remove(bi)
                        improved = True
                        found_improvement = True
                        break
                    else:
                        # Try to find 2 blocks that cover all uncovered_indices
                        uncovered_indices.sort(key=lambda x: len(t_to_blocks[x]))
                        
                        # Get candidate blocks for the most constrained uncovered subset
                        candidates_first = set(t_to_blocks[uncovered_indices[0]])
                        
                        # Filter: not in selected_set, not bi/bj/bl
                        valid_first = [bk for bk in candidates_first 
                                       if bk not in selected_set and bk != bi and bk != bj and bk != bl]
                        
                        if valid_first:
                            # For each candidate first block, check if a second block can cover the rest
                            found_pair = False
                            for bk1 in valid_first:
                                if time.time() > deadline - 10:
                                    break
                                
                                # Compute remaining uncovered after adding bk1
                                remaining = []
                                for idx in uncovered_indices:
                                    if idx not in block_t_indices[bk1]:
                                        remaining.append(idx)
                                
                                if not remaining:
                                    # bk1 alone covers all uncovered
                                    bk2 = None
                                else:
                                    # Find a second block that covers all remaining
                                    remaining.sort(key=lambda x: len(t_to_blocks[x]))
                                    candidates_second = set(t_to_blocks[remaining[0]])
                                    for ridx in remaining[1:]:
                                        candidates_second.intersection_update(t_to_blocks[ridx])
                                        if not candidates_second:
                                            break
                                    
                                    valid_second = [bk for bk in candidates_second 
                                                   if bk not in selected_set and bk != bi and bk != bj and bk != bl and bk != bk1]
                                    if valid_second:
                                        bk2 = random.choice(valid_second)
                                    else:
                                        bk2 = None
                                
                                if bk2 is not None or not remaining:
                                    # Remove bi, bj, bl
                                    current.remove(bl)
                                    current.remove(bj)
                                    current.remove(bi)
                                    selected_set.remove(bl)
                                    selected_set.remove(bj)
                                    selected_set.remove(bi)
                                    
                                    # Add bk1 (and bk2 if needed)
                                    current.append(bk1)
                                    selected_set.add(bk1)
                                    for idx in block_t_indices[bk1]:
                                        cover_count[idx] += 1
                                    
                                    if bk2 is not None:
                                        current.append(bk2)
                                        selected_set.add(bk2)
                                        for idx in block_t_indices[bk2]:
                                            cover_count[idx] += 1
                                    
                                    improved = True
                                    found_improvement = True
                                    break
                            
                            if found_pair:
                                break
                        else:
                            # No valid first block, restore counts
                            for idx in block_t_indices[bi]:
                                cover_count[idx] += 1
                            for idx in block_t_indices[bj]:
                                cover_count[idx] += 1
                            for idx in block_t_indices[bl]:
                                cover_count[idx] += 1
                        continue
                    
                    # Restore counts if we didn't find an improvement
                    if not found_improvement:
                        for idx in block_t_indices[bi]:
                            cover_count[idx] += 1
                        for idx in block_t_indices[bj]:
                            cover_count[idx] += 1
                        for idx in block_t_indices[bl]:
                            cover_count[idx] += 1
                
            if found_improvement:
                continue
        
        return current
>>>>>>> REPLACE

