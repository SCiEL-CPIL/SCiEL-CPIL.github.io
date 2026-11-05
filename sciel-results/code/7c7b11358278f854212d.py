import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # 1. Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Generate all k-blocks and their masks
    k_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for b in k_blocks:
        mask = 0
        # Get all t-subsets of b
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # 3. Greedy Construction
    full_mask = (1 << n_t) - 1
    covered_mask = 0
    selected_masks = []
    
    # To speed up, we can keep a list of active block indices?
    # Or just iterate all. Since we remove none during greedy, we iterate all.
    
    while covered_mask != full_mask:
        if time.time() > deadline:
            break # Return best found so far (invalid if not full, but better than exception)
            
        best_gain = -1
        best_block_idx = -1
        
        # Optimization: If best_gain reaches max possible (t-comb of k), break?
        # Max gain is comb(k, t).
        max_possible_gain = len(list(itertools.combinations(range(k), t))) # Just comb(k,t)
        
        for i, bm in enumerate(block_masks):
            gain = (bm & ~covered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if best_gain == max_possible_gain:
                    break # Can't do better
        
        if best_block_idx == -1:
            break # Should not happen if valid
            
        selected_masks.append(block_masks[best_block_idx])
        covered_mask |= block_masks[best_block_idx]
        
        if time.time() > deadline:
            break

    # 4. Redundancy Removal
    # We want to remove any block that is fully covered by the union of others.
    # We do this iteratively.
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        n_sel = len(selected_masks)
        for i in range(n_sel):
            if time.time() > deadline:
                break
            # Compute union of all others
            union_others = 0
            for j in range(n_sel):
                if i != j:
                    union_others |= selected_masks[j]
            
            # Check if selected_masks[i] is subset of union_others
            # i.e., selected_masks[i] & ~union_others == 0
            if (selected_masks[i] & ~union_others) == 0:
                selected_masks.pop(i)
                changed = True
                break # Restart loop
    
    # 5. Convert masks back to blocks
    # We need to map mask back to the original block tuple/list.
    # We stored block_masks in order of k_blocks.
    # We need to find which blocks correspond to the selected masks.
    # Since masks might not be unique? No, each block has a unique set of t-subsets?
    # Actually, different blocks can have the same set of t-subsets?
    # No, a block is defined by its k elements. The set of t-subsets uniquely identifies the block?
    # If two blocks have the same t-subsets, they must be the same block?
    # Yes, for k > t, the union of all t-subsets of a k-set is the k-set.
    # So we can map mask -> block.
    
    mask_to_block = {}
    for i, b in enumerate(k_blocks):
        mask = block_masks[i]
        # If multiple blocks have same mask, keep one?
        # They are equivalent for covering.
        if mask not in mask_to_block:
            mask_to_block[mask] = list(b)
            
    final_blocks = []
    for m in selected_masks:
        if m in mask_to_block:
            final_blocks.append(mask_to_block[m])
            
    return final_blocks
