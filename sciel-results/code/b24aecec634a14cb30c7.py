import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # Compute bitmask for each k-subset
    masks = [0] * n_blocks
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Greedy Set Cover
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * n_blocks
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_blocks):
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
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    best_solution = selected_greedy[:]
    current_best = len(best_solution)
    
    def try_remove_redundant(blocks_list):
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                # Check if block i is redundant
                # Union of all other blocks should cover all_mask
                union_others = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        union_others |= masks[blocks_list[j]]
                if union_others == all_mask:
                    blocks_list.pop(i)
                    removed = True
                    break
        return blocks_list
    
    best_found = try_remove_redundant(best_solution[:])
    current_best = len(best_found)
    
    start_time = time.time()
    
    # Multi-start with randomized greedy
    for restart in range(20):
        if time.time() - start_time > 90:
            break
        
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            # Find best gain
            best_gain = -1
            candidates = []
            for i in range(n_blocks):
                if not used[i]:
                    gain = (masks[i] & uncovered).bit_count()
                    if gain > 0:
                        candidates.append((gain, i))
            
            if not candidates:
                break
            
            # Sort by gain descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            max_gain = candidates[0][0]
            
            # Pick from top candidates with gain >= max_gain - 2
            top_candidates = [i for g, i in candidates if g >= max_gain - 2]
            
            if restart == 0:
                idx = top_candidates[0]
            else:
                # Randomized selection
                # Weight by gain
                weights = [(masks[i] & uncovered).bit_count() for i in top_candidates]
                total_w = sum(weights)
                if total_w == 0:
                    idx = top_candidates[0]
                else:
                    r = random.random() * total_w
                    cum = 0
                    for i, w in zip(top_candidates, weights):
                        cum += w
                        if r <= cum:
                            idx = i
                            break
                    else:
                        idx = top_candidates[-1]
            
            selected.append(idx)
            used[idx] = True
            uncovered &= ~masks[idx]
            
        selected = try_remove_redundant(selected)
        
        if len(selected) < current_best:
            current_best = len(selected)
            best_found = selected[:]
            
    # Local search: try to swap one block for another to enable removing a third
    # Or simply try random swaps
    for _ in range(50):
        if time.time() - start_time > 100:
            break
        
        if len(best_found) < 2:
            break
            
        # Pick two random indices in the solution
        i, j = random.sample(range(len(best_found)), 2)
        old_i, old_j = best_found[i], best_found[j]
        
        # Remove both
        remaining = [b for idx, b in enumerate(best_found) if idx != i and idx != j]
        
        # Check if remaining covers all
        union = 0
        for b in remaining:
            union |= masks[b]
        
        if union != all_mask:
            # Find t-subsets not covered
            gap = all_mask & ~union
            
            # Find a single block that covers the gap
            # To keep count same, we need to add 2 blocks? No, we removed 2, so we need to add <=2.
            # Ideally add 1 block that covers gap.
            found_block = -1
            # Find block with max gain on gap
            max_gain_gap = 0
            for b in range(n_blocks):
                if b not in remaining and b != old_i and b != old_j:
                    gain = (masks[b] & gap).bit_count()
                    if gain > max_gain_gap:
                        max_gain_gap = gain
                        found_block = b
            
            if max_gain_gap == gap.bit_count():
                # We can replace old_i, old_j with found_block? 
                # Wait, if we add found_block, count is len(remaining) + 1 = len(best_found) - 1.
                # This reduces count by 1!
                new_solution = remaining + [found_block]
                # Verify
                union_check = 0
                for b in new_solution:
                    union_check |= masks[b]
                if union_check == all_mask:
                    # Remove redundants just in case
                    new_solution = try_remove_redundant(new_solution)
                    if len(new_solution) < current_best:
                        current_best = len(new_solution)
                        best_found = new_solution[:]
        
        # Try replacing one block with another to potentially enable redundancy removal later?
        # Or just try single block swap: replace A with B. Count stays same.
        # This doesn't help minimize count directly unless B allows removing C.
        
    return [list(k_subsets[i]) for i in best_found]
