import itertools

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * len(k_subsets)
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # 4. Greedy Set Cover
    uncovered = all_mask
    selected_indices = []
    used_blocks = [False] * len(k_subsets)
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        
        for i in range(len(k_subsets)):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    # 5. Local Search: Remove redundant blocks
    # We iterate until no more blocks can be removed
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to remove items safely, or just mark for removal
        for idx_in_list in range(len(selected_indices) - 1, -1, -1):
            block_idx = selected_indices[idx_in_list]
            block_mask = masks[block_idx]
            
            # Check if the union of all other selected blocks covers everything
            # If yes, this block is redundant.
            # Optimization: If the block mask is a subset of the union of others, it's redundant.
            # Union of others = all_mask if covered.
            # We can compute union of others by OR-ing all masks except this one.
            
            # To avoid O(N^2) full recomputation, we can maintain a frequency count?
            # No, simple recomputation is fine for N~300.
            
            union_others = 0
            for other_idx in selected_indices:
                if other_idx != block_idx:
                    union_others |= masks[other_idx]
                    
            if union_others == all_mask:
                # Redundant, remove it
                selected_indices.pop(idx_in_list)
                changed = True
                # Restart the loop to check if more can be removed
                break
                
    # 6. Construct final list of blocks
    return [list(k_subsets[i]) for i in selected_indices]
