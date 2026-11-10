import itertools
import time
import random

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        return _simple_greedy(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precompute mapping for t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    combo_to_id = {c: i for i, c in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute masks for all k-subsets
    block_masks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << combo_to_id[sub])
        block_masks.append(mask)
        
    num_blocks = len(block_masks)
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple randomized greedy runs
    num_iterations = 30 # Increased iterations, but with optimization it's faster
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Shuffle indices to randomize tie-breaking
        indices = list(range(num_blocks))
        random.shuffle(indices)
        
        uncovered = (1 << num_t_subsets) - 1
        selected_indices = []
        
        # Active list optimization
        active_indices = indices[:]
        
        while uncovered and active_indices:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            
            # Find best in active list
            for idx in active_indices:
                m = block_masks[idx]
                inter = m & uncovered
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        best_mask = m
                        if gain == num_t_subsets:
                            break
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            uncovered &= ~best_mask
            
            # Update active list: remove blocks that no longer cover anything new
            # This is O(N_active)
            new_active = []
            for idx in active_indices:
                if block_masks[idx] & uncovered:
                    new_active.append(idx)
            active_indices = new_active
            
        # Prune
        selected_indices = _prune_blocks(selected_indices, block_masks)
        
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_blocks = selected_indices[:]
            
    if best_blocks is None:
        return _simple_greedy(v, k, t)
        
    all_blocks = list(itertools.combinations(range(v), k))
    result_blocks = [list(all_blocks[i]) for i in best_blocks]
    return result_blocks

def _prune_blocks(selected_indices, block_masks):
    selected_indices = selected_indices[:]
    changed = True
    while changed:
        changed = False
        new_selected = []
        n = len(selected_indices)
        for i in range(n):
            idx = selected_indices[i]
            mask_i = block_masks[idx]
            union_others = 0
            for j in range(n):
                if j != i:
                    union_others |= block_masks[selected_indices[j]]
            
            if (mask_i & ~union_others) == 0:
                changed = True
            else:
                new_selected.append(idx)
        selected_indices = new_selected
    return selected_indices

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        best_block = None
        best_gain = -1
        for block in itertools.combinations(range(v), k):
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        if best_gain <= 0:
            break
        blocks.append(list(best_block))
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks
