import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Map all t-subsets to indices
    t_combos = list(itertools.combinations(range(v), t))
    combo_to_idx = {c: i for i, c in enumerate(t_combos)}
    n_t_subsets = len(t_combos)
    full_mask = (1 << n_t_subsets) - 1
    
    # 2. Precompute coverage masks for all k-subsets
    k_combos = list(itertools.combinations(range(v), k))
    masks = []
    for block in k_combos:
        mask = 0
        # For each t-subset within this block, set the corresponding bit
        for sub in itertools.combinations(block, t):
            idx = combo_to_idx[sub]
            mask |= (1 << idx)
        masks.append(mask)
        
    # 3. Global Greedy Set Cover
    covered_mask = 0
    blocks = []
    # List of indices of blocks that are still candidates
    candidates = list(range(len(k_combos)))
    
    while covered_mask != full_mask:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_idx = -1
        
        # Find the block that covers the most uncovered t-subsets
        for idx in candidates:
            # Number of newly covered subsets
            gain = bin(masks[idx] & ~covered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if gain == 0:
                    # No block can cover more, but we must have uncovered ones?
                    # If gain is 0 for all, we are stuck or done. 
                    # But if covered_mask != full_mask, there must be a block with gain > 0 
                    # unless the problem is infeasible (which it isn't).
                    pass
        
        if best_idx == -1:
            # Should not happen for valid covering problems
            break
            
        # Select the block
        blocks.append(list(k_combos[best_idx]))
        covered_mask |= masks[best_idx]
        
        # Remove selected block from candidates to avoid re-selection
        # Since blocks must be distinct, we don't want to pick the same one again.
        # Removing it from the list is O(N). We can just filter or swap-pop.
        # For simplicity and since N is ~50k, filtering is okay but might be slow if done 150 times.
        # 150 * 50k = 7.5M operations for list reconstruction. Acceptable.
        # Or we can keep a "used" boolean array and skip.
        # Let's use a set of used indices for O(1) check, or just remove from list.
        # Removing from list:
        candidates.remove(best_idx)
        
    return blocks
