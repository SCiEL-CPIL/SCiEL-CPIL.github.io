import itertools
import random
import time

def construct(v, k, t):
    # Parameters
    # v=24, k=7, t=3
    
    # 1. Precompute all blocks and their masks
    # Map each t-subset to an index
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)
    
    # Precompute mask for each k-block
    # mask is an integer where bit i is set if t_subset_to_idx[i] is in the block
    block_masks = [0] * num_blocks
    block_to_indices = {} # map block tuple to index in all_k_blocks for quick lookup if needed
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
        block_to_indices[block] = i
        
    # Full mask: all bits set
    full_mask = (1 << num_t_subsets) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Random seed for reproducibility in evaluation? No, evaluator might run multiple times.
    # But we want variety.
    
    # Run multiple greedy attempts
    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1
        
        # Greedy Construction
        uncovered = full_mask
        current_blocks = []
        
        while uncovered:
            # Find block with max gain
            max_gain = -1
            best_block_idx = -1
            best_gain_val = 0
            
            # To speed up, we can iterate through all blocks
            # num_blocks is ~346k. 
            # popcount of (uncovered & mask)
            # Python int bit_count is fast.
            
            # Optimization: Only consider blocks that have gain > 0
            # We can't easily filter without iterating.
            
            # Iterate
            for i in range(num_blocks):
                m = block_masks[i]
                # If m is 0, skip (shouldn't happen)
                # gain = popcount(uncovered & m)
                gain = (uncovered & m).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
                    best_gain_val = gain
                    if max_gain == 35: # Max possible gain for k=7, t=3 is C(7,3)=35
                        # Can't do better than 35
                        # But there might be multiple. Break? 
                        # No, we want random tie-breaking.
                        # So we don't break immediately, or we collect all with max gain.
                        pass
            
            if best_block_idx == -1:
                break # Should not happen if uncovered != 0
                
            current_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # Redundancy Removal
        # Convert to set of indices for easier manipulation
        block_set = set(current_blocks)
        
        changed = True
        while changed:
            changed = False
            for b_idx in list(block_set):
                # Check if b_idx is redundant
                # Union of all other blocks
                # Efficient way: 
                # Count how many times each t-subset is covered by blocks in block_set
                # If all t-subsets in b_idx are covered by others, remove b_idx
                
                # This O(N^2) check might be slow if N is large, but N~80.
                # 80 * 80 * 35 ops = ~200k ops. Fast.
                
                # Let's use a counting array for t-subsets
                # But doing it from scratch each time is O(N * K * T)
                # N=80, K=7, T=3. 
                # Better: Maintain coverage counts incrementally? 
                # For simplicity, just recompute for the current set.
                
                # Optimization: 
                # Precompute list of t-subset indices for each block
                block_t_idx_lists = []
                for b in block_set:
                    # We need the t-subsets of this block
                    # We can store this precomputed too
                    pass
                
                # Let's precompute t_idx_list for all blocks
                # block_t_indices[i] = list of t-subset indices covered by block i
                
                # Actually, we can just check:
                # Is there any t-subset in block b_idx that is ONLY covered by b_idx in block_set?
                
                # To do this efficiently:
                # Create a counter array of size num_t_subsets
                # Iterate all blocks in block_set, increment count for their t-subsets
                # Then for b_idx, check if any of its t-subsets have count == 1
                # If yes, it's not redundant.
                # If no, it is redundant.
                
                # This is O(|block_set| * C(k,t))
                # 80 * 35 = 2800 increments.
                # Then check 35 items.
                # Total ~3000 ops per block.
                # 80 blocks * 3000 = 240k ops per pass.
                # Very fast.
                
                # Implementation:
                counts = [0] * num_t_subsets
                for b in block_set:
                    # Get t-subset indices for b
                    # We need to map block index to its t-subset indices
                    # Let's precompute this list of lists
                    pass
                
                # Let's precompute `block_t_indices`
                # This takes memory but is fast.
                
                # If redundant, remove
                # ...
                
            # End while changed
            
        # End Redundancy Removal
        
        if len(block_set) < best_count:
            best_count = len(block_set)
            best_blocks = list(block_set)
            
        # Check time again
        if time.time() - start_time > timeout - 5:
            break
            
    # Convert indices back to blocks
    if best_blocks is None:
        # Fallback: simple greedy
        # ...
        pass
        
    return [list(all_k_blocks[i]) for i in best_blocks]
