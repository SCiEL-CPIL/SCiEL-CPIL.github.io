import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Precompute masks for all k-subsets (blocks)
    # A mask is a Python integer where bit i is 1 if the i-th t-subset is contained in the block
    k_subsets = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in k_subsets:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # Map mask to the block (for reconstruction)
    # Since blocks are distinct, we can just keep the index
    # We need to return the actual points
    
    # 3. Greedy Set Cover
    # uncovered: bit i is 1 if t-subset i is not yet covered
    uncovered = (1 << num_t_subsets) - 1
    selected_indices = []
    
    # Optimization: Sort blocks initially? No, greedy picks dynamically.
    # We will iterate through all blocks to find the best one each time.
    
    n_blocks = len(k_subsets)
    
    while uncovered:
        # Check time
        if time.time() > deadline:
            # Return current best? 
            # The problem requires a valid covering. 
            # If we run out of time, we might return an invalid covering.
            # However, greedy usually finishes well before 120s.
            # If we hit deadline, we should probably break and return what we have, 
            # but it might be invalid. 
            # The prompt says "Valid when every t-subset... lies in at least one block".
            # If we return an invalid one, score is 0.
            # Given the speed, we are unlikely to hit this.
            pass
            
        best_gain = 0
        best_idx = -1
        
        # Find the block that covers the most uncovered t-subsets
        for i in range(n_blocks):
            mask = block_masks[i]
            # Calculate marginal gain
            gain = (uncovered & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            # Early break if we covered everything? 
            # If gain == num_uncovered, we can't do better.
            # But we don't track num_uncovered easily in the loop.
        
        if best_gain == 0:
            # Should not happen if a covering exists
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
        # Optional: Remove selected block from consideration? 
        # It's already covered, so its gain will be 0 in future iterations anyway.
        # No need to remove, just the gain calculation will yield 0.
        
    # Convert indices back to blocks
    result = [list(k_subsets[i]) for i in selected_indices]
    return result
