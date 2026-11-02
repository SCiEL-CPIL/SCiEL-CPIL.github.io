    def add_remove_local_search(initial_blocks, counts):
        """
        Perform Add-Remove local search to minimize block count.
        """
        current_blocks = list(initial_blocks)
        current_counts = list(counts)
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(42)
        
        # Helper to try removing blocks greedily
        def try_remove_blocks(blocks, counts):
            removed = True
            while removed:
                removed = False
                # Shuffle indices to randomize order
                indices = list(range(len(blocks)))
                rng.shuffle(indices)
                
                for idx in indices:
                    if elapsed() > deadline - 2:
                        return False # Timeout
                    
                    b = blocks[idx]
                    # Check if we can remove b
                    can_remove = True
                    for ts in itertools.combinations(b, t):
                        ts_idx = tsubset_index[ts]
                        if counts[ts_idx] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        # Remove it
                        blocks.pop(idx)
                        for ts in itertools.combinations(b, t):
                            counts[tsubset_index[ts]] -= 1
                        removed = True
                        break # Restart loop with new list
                
            return True # Finished without timeout

        # Main loop
        while elapsed() < deadline - 5:
            # 1. Try to remove blocks
            try_remove_blocks(current_blocks, current_counts)
            
            # 2. Update best
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)
            
            # 3. If we can't remove any more (or just to escape local min), add a block
            # Heuristic: Add a block that covers many t-subsets with count == 1
            # This creates redundancy, allowing future deletions.
            
            # To speed up, we can just add a random block if time is tight,
            # or a "smart" block.
            
            # Smart Add: Find a block that maximizes coverage of count=1 subsets.
            # This is expensive if done from scratch.
            # Alternative: Just add a random block. It's simple and often works with enough iterations.
            # Let's try a "semi-smart" add: Pick a random t-subset with count==1, 
            # and extend it to a k-block greedily? 
            # Or just random block.
            
            # Let's try random block first for simplicity and speed.
            if len(current_blocks) > 0:
                # Generate a random block
                new_block = list(rng.sample(range(v), k))
                new_block.sort()
                
                # Check if it's already in the list (unlikely but possible)
                # Using a set for membership check is O(1) if we maintain it, but list is O(N).
                # N is ~300, so O(N) is fine.
                if new_block not in current_blocks:
                    current_blocks.append(new_block)
                    for ts in itertools.combinations(new_block, t):
                        current_counts[tsubset_index[ts]] += 1
                    
                    # Immediately try to remove blocks again
                    try_remove_blocks(current_blocks, current_counts)
                    
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)

        return best_blocks
