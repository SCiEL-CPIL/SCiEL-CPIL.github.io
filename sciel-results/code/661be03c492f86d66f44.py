<<<<<<< SEARCH
    def local_search(blocks, seed=0, max_iterations=200):
        """Intensive local search: iterative removal + targeted addition."""
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        reduced = list(blocks)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
                
            # Phase 1: Remove redundant blocks
            reduced = remove_redundant(reduced)
            
            # Phase 2: Try to swap blocks to reduce count
            if len(reduced) <= 1:
                break
                
            # Build cover counts
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsub_lists = []
            for b in reduced:
                tsubs = block_tsubs[b].tolist()
                block_tsub_lists.append(tsubs)
                for idx in tsubs:
                    cover_count[idx] += 1
                    
            improved = False
            n = len(reduced)
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2:
                    break
                if len(reduced) <= 1:
                    break
                    
                # Find t-subsets uniquely covered by block i
                unique_tsubs = [idx for idx in block_tsub_lists[i] if cover_count[idx] == 1]
                if not unique_tsubs:
                    # Block is redundant, remove it
                    for idx in block_tsub_lists[i]:
                        cover_count[idx] -= 1
                    block_tsub_lists.pop(i)
                    reduced.pop(i)
                    improved = True
                    continue
                    
                # Try to find a replacement block that covers all unique_tsubs
                candidate_set = set()
                for ts_idx in unique_tsubs:
                    for bi in tsub_blocks[ts_idx]:
                        # Check if bi is not already in reduced
                        # This check is O(n) if done naively. 
                        # For speed, we can skip this check initially and verify later,
                        # or use a set of current block indices.
                        if bi not in reduced:
                            candidate_set.add(bi)
                            
                if not candidate_set:
                    continue
                    
                candidates = list(candidate_set)
                rng.shuffle(candidates)
                
                found_swap = False
                for ci in candidates[:50]:  # Limit candidates tried
                    if time.time() > deadline - 2:
                        break
                        
                    cand_tsubs_set = set(block_tsubs[ci].tolist())
                    
                    # Check if candidate covers all unique t-subsets of block i
                    if not all(ts in cand_tsubs_set for ts in unique_tsubs):
                        continue
                        
                    # Check: after removing block i and adding ci, no t-subset is uncovered
                    ok = True
                    for idx in block_tsub_lists[i]:
                        if cover_count[idx] == 1 and idx not in cand_tsubs_set:
                            ok = False
                            break
                            
                    if ok:
                        # Perform swap
                        for idx in block_tsub_lists[i]:
                            cover_count[idx] -= 1
                        for ts_idx in block_tsubs[ci]:
                            cover_count[ts_idx] += 1
                        reduced[i] = ci
                        block_tsub_lists[i] = block_tsubs[ci].tolist()
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    continue
                    
            if not improved:
                # Try adding a block that covers many currently singly-covered t-subsets
                if len(reduced) > 2 and time.time() < deadline - 5:
                    single_covered = np.where(cover_count == 1)[0]
                    if len(single_covered) > 0:
                        # Find block that covers the most single-covered t-subsets
                        best_gain = -1
                        best_block = -1
                        for bi in range(n_blocks):
                            if bi in reduced:
                                continue
                            cnt = 0
                            for ts_idx in block_tsubs[bi]:
                                if cover_count[ts_idx] == 1:
                                    cnt += 1
                            if cnt > best_gain:
                                best_gain = cnt
                                best_block = bi
                                
                        if best_gain >= 2 and best_block >= 0:
                            # Add this block, then try to remove a redundant one
                            new_reduced = reduced + [best_block]
                            new_reduced = remove_redundant(new_reduced)
                            if len(new_reduced) < len(reduced):
                                reduced = new_reduced
                                improved = True
                                
            if not improved:
                break
                
        return reduced
=======
    def local_search(blocks, seed=0, max_iterations=200):
        """Intensive local search: iterative removal + targeted addition."""
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        reduced = list(blocks)
        
        # Precompute block t-subset sets for fast membership testing
        # Only for blocks that might be candidates (not in initial solution)
        # We'll compute lazily and cache
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
                
            # Phase 1: Remove redundant blocks
            reduced = remove_redundant(reduced)
            
            # Phase 2: Try to swap blocks to reduce count
            if len(reduced) <= 1:
                break
                
            # Build cover counts and set of current block indices for O(1) lookup
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsub_lists = []
            reduced_set = set(reduced)
            for b in reduced:
                tsubs = block_tsubs[b].tolist()
                block_tsub_lists.append(tsubs)
                for idx in tsubs:
                    cover_count[idx] += 1
                    
            improved = False
            n = len(reduced)
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2:
                    break
                if len(reduced) <= 1:
                    break
                    
                # Find t-subsets uniquely covered by block i
                unique_tsubs = [idx for idx in block_tsub_lists[i] if cover_count[idx] == 1]
                if not unique_tsubs:
                    # Block is redundant, remove it
                    for idx in block_tsub_lists[i]:
                        cover_count[idx] -= 1
                    block_tsub_lists.pop(i)
                    reduced.pop(i)
                    reduced_set.discard(i)
                    improved = True
                    continue
                    
                # Try to find a replacement block that covers all unique_tsubs
                # Use intersection of blocks covering each unique t-subset
                candidate_set = None
                for ts_idx in unique_tsubs:
                    blocks_covering = set(tsub_blocks[ts_idx].tolist())
                    blocks_covering -= reduced_set  # Remove blocks already in solution
                    if candidate_set is None:
                        candidate_set = blocks_covering
                    else:
                        candidate_set &= blocks_covering
                    if not candidate_set:
                        break
                        
                if not candidate_set:
                    continue
                    
                candidates = list(candidate_set)
                rng.shuffle(candidates)
                
                found_swap = False
                for ci in candidates[:50]:  # Limit candidates tried
                    if time.time() > deadline - 2:
                        break
                        
                    # Fast check: does candidate cover all unique t-subsets?
                    # Since ci came from intersection, it should by construction
                    # But verify to be safe
                    cand_tsubs = block_tsubs[ci]
                    
                    # Check: after removing block i and adding ci, no t-subset is uncovered
                    # Only need to check t-subsets in block i that have cover_count == 1
                    # and are NOT in candidate
                    ok = True
                    for idx in block_tsub_lists[i]:
                        if cover_count[idx] == 1:
                            # Check if idx is in candidate
                            # Use np.isin for speed
                            if not np.any(cand_tsubs == idx):
                                ok = False
                                break
                            
                    if ok:
                        # Perform swap
                        for idx in block_tsub_lists[i]:
                            cover_count[idx] -= 1
                        for ts_idx in cand_tsubs:
                            cover_count[ts_idx] += 1
                        reduced[i] = ci
                        block_tsub_lists[i] = cand_tsubs.tolist()
                        reduced_set.discard(i)
                        reduced_set.add(ci)
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    continue
                    
            if not improved:
                # Try adding a block that covers many currently singly-covered t-subsets
                if len(reduced) > 2 and time.time() < deadline - 5:
                    single_covered_mask = (cover_count == 1)
                    if np.any(single_covered_mask):
                        # Find block that covers the most single-covered t-subsets
                        best_gain = -1
                        best_block = -1
                        for bi in range(n_blocks):
                            if bi in reduced_set:
                                continue
                            # Count single-covered t-subsets in this block
                            cnt = np.sum(single_covered_mask[block_tsubs[bi]])
                            if cnt > best_gain:
                                best_gain = cnt
                                best_block = bi
                                
                        if best_gain >= 2 and best_block >= 0:
                            # Add this block, then try to remove a redundant one
                            new_reduced = reduced + [best_block]
                            new_reduced = remove_redundant(new_reduced)
                            if len(new_reduced) < len(reduced):
                                reduced = new_reduced
                                improved = True
                                
            if not improved:
                break
                
        return reduced
>>>>>>> REPLACE
