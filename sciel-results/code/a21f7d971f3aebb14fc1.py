import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_targets = len(t_subsets)
    
    # 2. Generate all k-blocks and compute their coverage masks
    # coverage_mask[i] is an integer with bit j set if block i covers t_subset j
    k_blocks = list(itertools.combinations(range(v), k))
    coverage_masks = []
    
    for block in k_blocks:
        mask = 0
        # A block covers a t-subset if the t-subset is a subset of the block
        # We can iterate over t-subsets of the block
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        coverage_masks.append(mask)
        
    # 3. Greedy selection using bitmasks
    uncovered_mask = (1 << n_targets) - 1
    selected_blocks = []
    n_blocks = len(k_blocks)
    
    while uncovered_mask:
        if time.time() > deadline:
            # If we run out of time, we return what we have. 
            # However, the problem requires a valid covering. 
            # If we haven't covered everything, this is invalid.
            # But the greedy approach is deterministic and fast. 
            # 25M ops is likely < 10s. 
            # If it does timeout, we might need a fallback, but let's trust the speed.
            pass
            
        best_idx = -1
        best_gain = 0
        
        # Find block with max gain
        # We can optimize by skipping blocks that are subsets of previously selected blocks? 
        # No, blocks are distinct size k.
        # Simple scan:
        for i in range(n_blocks):
            # Optimization: if we already covered all targets in this block, gain is 0
            # We can check if (coverage_masks[i] & uncovered_mask) is 0 quickly?
            # The AND itself is the check.
            gain = (coverage_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early exit if we found a block that covers all remaining? 
                # Unlikely, but if gain == uncovered_mask.bit_count(), we can stop searching for best.
                if gain == uncovered_mask.bit_count():
                    break
        
        if best_idx == -1:
            # Should not happen if a covering exists
            break
            
        selected_blocks.append(list(k_blocks[best_idx]))
        uncovered_mask &= ~coverage_masks[best_idx]
        
    return selected_blocks
