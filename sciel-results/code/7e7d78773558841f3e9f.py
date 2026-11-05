import itertools
import time

def construct(v, k, t):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10 seconds margin from 120s limit
    
    # 1. Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute all k-subsets and their coverage masks
    # A k-subset covers a t-subset if the t-subset is a subset of the k-subset
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_subset_masks = []
    for k_sub in all_k_subsets:
        mask = 0
        # For a k-subset, generate all t-subsets within it
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        k_subset_masks.append(mask)
        
    # 3. Greedy Set Cover
    # uncovered_mask starts with all bits set
    uncovered_mask = (1 << num_t_subsets) - 1
    selected_blocks = []
    
    # To avoid re-selecting the same block, we can track used indices, 
    # but since coverage becomes 0, it won't be picked again if we strictly maximize gain > 0.
    # However, if gain is 0 for all, we are stuck (shouldn't happen if valid cover exists).
    
    while uncovered_mask != 0:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        # Scan all k-subsets to find the one covering the most uncovered t-subsets
        # Optimization: We can skip blocks that we've already selected? 
        # No, because we need to find the *best* remaining coverage.
        # Actually, once a block is selected, its contribution to 'uncovered' is 0.
        # So we can just scan all.
        
        for i, mask in enumerate(k_subset_masks):
            # Count bits in intersection
            # In Python 3.8+, int.bit_count() is available
            gain = (uncovered_mask & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1 or best_gain == 0:
            # No progress possible? This shouldn't happen for valid parameters
            break
            
        # Select the best block
        best_block = all_k_subsets[best_block_idx]
        selected_blocks.append(list(best_block))
        
        # Update uncovered mask
        uncovered_mask &= ~k_subset_masks[best_block_idx]
        
    return selected_blocks
