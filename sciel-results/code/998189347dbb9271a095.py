<<<<<<< SEARCH
    def local_search(selected_blocks, counts, seed_val, max_iters=500):
        """
        Try to swap blocks to reduce the count further.
        1. Try to remove a block. If it breaks coverage, try to find a replacement.
        2. If no replacement found, try to swap with a block that covers more unique parts?
        Actually, just trying to remove is the most direct path to reduction.
        If we can't remove, we can try to swap to create redundancy elsewhere.
        """
        random.seed(seed_val)
        n_sel = len(selected_blocks)
        if n_sel == 0:
            return selected_blocks, counts

        # We work with a copy
        sel_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random block to try to remove
            rem_pos = random.randint(0, len(sel_blocks) - 1)
            rem_idx = sel_blocks[rem_pos]
            rem_ts = block_t_indices_np[rem_idx]
            
            # Check if removing it leaves any t-subset uncovered
            # i.e., if any count in rem_ts is 1
            if np.any(cur_counts[rem_ts] == 1):
                # It would break coverage.
                # We can try to swap: remove rem_idx, add a new_idx that covers the broken ones.
                broken_ts = rem_ts[cur_counts[rem_ts] == 1]
                
                # Find a block that covers the most broken_ts
                # Candidate blocks: all blocks that cover at least one broken_ts.
                # We can precompute a map from t-subset index to list of blocks covering it?
                # That's M.T.
                # Let's use M.T to find candidates.
                
                # Get indices of blocks that cover any of the broken_ts
                # M.T is num_t_subs x num_blocks
                # We want union of rows corresponding to broken_ts
                
                # To avoid full matrix multiplication, we can just check candidates.
                # But checking 75k blocks is slow.
                # Let's sample.
                
                best_add = -1
                best_gain = 0
                best_total_gain = 0 # Total uncovered covered by this block? No, just broken.
                
                # Sample 200 random blocks
                # We can be smarter: sample from blocks that cover at least one broken_ts
                # But getting that list requires M.T operation.
                # M.T.dot(broken_mask) gives counts of broken_ts covered by each block.
                # Let's try that.
                
                broken_mask = np.zeros(num_t_subs, dtype=np.int32)
                broken_mask[broken_ts] = 1
                
                # This is a sparse matrix-vector product
                # M.T is (num_t_subs, num_blocks)
                # M.T.dot(broken_mask) -> (num_blocks,)
                # This tells us how many broken_ts each block covers.
                
                # This is expensive if done every iteration.
                # Let's only do it if broken_ts is small?
                # Or just sample.
                
                # Let's try sampling first. It's simpler.
                candidates = random.sample(range(num_blocks), min(200, num_blocks))
                
                for b_idx in candidates:
                    if b_idx in sel_blocks:
                        continue
                    add_ts_set = block_t_sets[b_idx]
                    gain = 0
                    for ts in broken_ts:
                        if ts in add_ts_set:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_add = b_idx
                        if gain == len(broken_ts):
                            break # Perfect cover for the broken parts
                
                if best_add != -1 and best_gain == len(broken_ts):
                    # Perform swap
                    # Remove rem_idx
                    cur_counts[rem_ts] -= 1
                    # Add best_add
                    add_ts = block_t_indices_np[best_add]
                    cur_counts[add_ts] += 1
                    
                    # Update list
                    sel_blocks[rem_pos] = best_add
                    # This swap maintains the same number of blocks.
                    # It might enable further reductions.
                    
            else:
                # It doesn't break coverage.
                # Remove it.
                cur_counts[rem_ts] -= 1
                sel_blocks.pop(rem_pos)
                # Since we removed a block, we might have made other blocks redundant.
                # The next iteration might catch it.
                
        return sel_blocks, cur_counts
=======
    def local_search(selected_blocks, counts, seed_val, max_iters=500):
        """
        Try to swap blocks to reduce the count further.
        1. Try to remove a block. If it breaks coverage, try to find a replacement.
        2. If no replacement found, try to swap with a block that covers more unique parts?
        Actually, just trying to remove is the most direct path to reduction.
        If we can't remove, we can try to swap to create redundancy elsewhere.
        """
        random.seed(seed_val)
        n_sel = len(selected_blocks)
        if n_sel == 0:
            return selected_blocks, counts

        # We work with a copy
        sel_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        
        # Build a set for O(1) membership check
        sel_block_set = set(sel_blocks)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random block to try to remove
            rem_pos = random.randint(0, len(sel_blocks) - 1)
            rem_idx = sel_blocks[rem_pos]
            rem_ts = block_t_indices_np[rem_idx]
            
            # Check if removing it leaves any t-subset uncovered
            # i.e., if any count in rem_ts is 1
            if np.any(cur_counts[rem_ts] == 1):
                # It would break coverage.
                # We can try to swap: remove rem_idx, add a new_idx that covers the broken ones.
                broken_ts = rem_ts[cur_counts[rem_ts] == 1]
                n_broken = len(broken_ts)
                
                if n_broken == 0:
                    # No broken subsets, so we can just remove
                    cur_counts[rem_ts] -= 1
                    sel_blocks.pop(rem_pos)
                    sel_block_set.discard(rem_idx)
                    continue
                
                # Find candidate blocks that cover at least one broken t-subset
                # Use M.T to get blocks covering each broken t-subset
                # M.T is (num_t_subs, num_blocks)
                # For each broken t-subset, get the blocks that cover it
                
                # Collect candidate blocks (union of blocks covering any broken t-subset)
                candidate_set = set()
                for ts_idx in broken_ts:
                    # Get row from M.T for this t-subset
                    # M.T has shape (num_t_subs, num_blocks)
                    # Row ts_idx gives which blocks cover this t-subset
                    row_start = M.T.indptr[ts_idx]
                    row_end = M.T.indptr[ts_idx + 1]
                    for col in M.T.indices[row_start:row_end]:
                        candidate_set.add(int(col))
                
                # Remove already selected blocks from candidates
                candidates = [b for b in candidate_set if b not in sel_block_set]
                
                if not candidates:
                    continue
                
                # Score each candidate: how many broken t-subsets it covers
                # Also consider how many currently-covered t-subsets it adds (to avoid making things worse)
                best_add = -1
                best_score = -1
                
                # Limit candidates to top few for speed
                if len(candidates) > 500:
                    # Sample 500 candidates
                    candidates = random.sample(candidates, 500)
                
                for b_idx in candidates:
                    add_ts = block_t_indices_np[b_idx]
                    add_ts_set = block_t_sets[b_idx]
                    
                    # Count how many broken t-subsets this block covers
                    n_covers_broken = sum(1 for ts in broken_ts if ts in add_ts_set)
                    
                    # Count how many t-subsets this block adds that are currently covered (redundancy)
                    # This is a penalty - we don't want to add too much redundancy
                    n_redundant = 0
                    for ts in add_ts:
                        if ts not in set(broken_ts) and cur_counts[ts] > 0:
                            n_redundant += 1
                    
                    # Score: maximize covered broken, minimize redundant additions
                    # Only accept if we cover all broken t-subsets
                    if n_covers_broken == n_broken:
                        score = n_covers_broken - 0.1 * n_redundant
                    else:
                        continue
                    
                    if score > best_score:
                        best_score = score
                        best_add = b_idx
                
                if best_add != -1:
                    # Perform swap
                    # Remove rem_idx
                    cur_counts[rem_ts] -= 1
                    sel_block_set.discard(rem_idx)
                    # Add best_add
                    add_ts = block_t_indices_np[best_add]
                    cur_counts[add_ts] += 1
                    sel_block_set.add(best_add)
                    
                    # Update list
                    sel_blocks[rem_pos] = best_add
                    # This swap maintains the same number of blocks.
                    # It might enable further reductions.
                    
            else:
                # It doesn't break coverage.
                # Remove it.
                cur_counts[rem_ts] -= 1
                sel_blocks.pop(rem_pos)
                sel_block_set.discard(rem_idx)
                # Since we removed a block, we might have made other blocks redundant.
                # The next iteration might catch it.
                
        return sel_blocks, cur_counts
>>>>>>> REPLACE

