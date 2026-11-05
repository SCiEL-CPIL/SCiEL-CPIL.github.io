<<<<<<< SEARCH
    def local_search(blocks, random_seed, max_iterations=100):
        """Try to improve the covering by replacing blocks."""
        rng = np.random.RandomState(random_seed)
        current_blocks = list(blocks)
        current_len = len(current_blocks)
        
        if current_len == 0:
            return current_blocks
        
        # Build coverage count
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for block in current_blocks:
            for sub in itertools.combinations(block, t):
                triple_count[triple_to_idx[sub]] += 1
        
        for _ in range(max_iterations):
            # Pick a random block to try removing
            if not current_blocks:
                break
            idx = rng.randint(len(current_blocks))
            candidate_block = current_blocks[idx]
            
            # Temporarily remove and check if coverage is still valid
            for sub in itertools.combinations(candidate_block, t):
                triple_count[triple_to_idx[sub]] -= 1
            
            # Check if all triples are still covered
            if np.min(triple_count) >= 1:
                # Removal is valid
                current_blocks.pop(idx)
                current_len -= 1
            else:
                # Restore
                for sub in itertools.combinations(candidate_block, t):
                    triple_count[triple_to_idx[sub]] += 1
            
            # Try to add a block that covers the most uncovered triples
            if np.any(triple_count == 0):
                # Find blocks that cover at least one uncovered triple
                gains = np.sum((triple_count == 0)[block_triple_indices], axis=1)
                max_gain = np.max(gains)
                if max_gain > 0:
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = candidates[rng.randint(len(candidates))]
                    new_block = all_blocks[best_idx]
                    
                    # Check if this block is already in current_blocks
                    if new_block not in current_blocks:
                        current_blocks.append(new_block)
                        current_len += 1
                        for sub in itertools.combinations(new_block, t):
                            triple_count[triple_to_idx[sub]] += 1
        
        return current_blocks
=======
    def local_search(blocks, random_seed, max_iterations=100):
        """Try to improve the covering by removing redundant blocks and swapping."""
        rng = np.random.RandomState(random_seed)
        current_blocks = list(blocks)
        current_len = len(current_blocks)
        
        if current_len == 0:
            return current_blocks
        
        # Build coverage count
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for block in current_blocks:
            for sub in itertools.combinations(block, t):
                triple_count[triple_to_idx[sub]] += 1
        
        # Build a set for fast membership check
        block_set = set(current_blocks)
        
        for iteration in range(max_iterations):
            # Pick a random block to try removing
            if not current_blocks:
                break
            idx = rng.randint(len(current_blocks))
            candidate_block = current_blocks[idx]
            
            # Temporarily remove and check if coverage is still valid
            for sub in itertools.combinations(candidate_block, t):
                triple_count[triple_to_idx[sub]] -= 1
            
            # Check if all triples are still covered
            if np.min(triple_count) >= 1:
                # Removal is valid
                current_blocks.pop(idx)
                current_len -= 1
                block_set.discard(candidate_block)
            else:
                # Restoration needed: try swap - remove candidate, add two new blocks
                # First, find the best single block to add
                gains = np.sum((triple_count == 0)[block_triple_indices], axis=1)
                max_gain = np.max(gains)
                
                if max_gain > 0:
                    # Try adding one block that covers the most uncovered triples
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = candidates[rng.randint(len(candidates))]
                    new_block = all_blocks[best_idx]
                    
                    if new_block not in block_set:
                        # Add it and see if we can then remove the candidate
                        for sub in itertools.combinations(new_block, t):
                            triple_count[triple_to_idx[sub]] += 1
                        
                        # Now check if candidate can be removed
                        for sub in itertools.combinations(candidate_block, t):
                            triple_count[triple_to_idx[sub]] -= 1
                        
                        if np.min(triple_count) >= 1:
                            # Swap successful: remove candidate, keep new_block
                            current_blocks.pop(idx)
                            current_len -= 1
                            block_set.discard(candidate_block)
                            current_blocks.append(new_block)
                            block_set.add(new_block)
                            # Note: triple_count already reflects the swap
                        else:
                            # Swap failed: restore everything
                            for sub in itertools.combinations(candidate_block, t):
                                triple_count[triple_to_idx[sub]] += 1
                            for sub in itertools.combinations(new_block, t):
                                triple_count[triple_to_idx[sub]] -= 1
                    else:
                        # New block already exists, restore
                        for sub in itertools.combinations(candidate_block, t):
                            triple_count[triple_to_idx[sub]] += 1
                else:
                    # No improvement possible, restore
                    for sub in itertools.combinations(candidate_block, t):
                        triple_count[triple_to_idx[sub]] += 1
        
        return current_blocks
>>>>>>> REPLACE

