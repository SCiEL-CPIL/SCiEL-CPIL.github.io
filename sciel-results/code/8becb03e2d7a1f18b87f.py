import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_blocks
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Precompute: for each t-subset index, which blocks cover it?
    # This helps in B&B to find candidates quickly.
    t_to_blocks = [[] for _ in range(n_t_subsets)]
    for i in range(n_blocks):
        m = masks[i]
        while m:
            bit = m & -m
            idx = bit.bit_length() - 1
            t_to_blocks[idx].append(i)
            m ^= bit

    # 4. Greedy Set Cover
    def run_greedy(randomize=False, top_k=1):
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            for i in range(n_blocks):
                if used[i]:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
            
            if not best_indices:
                break
            
            if randomize and len(best_indices) > 1:
                # Pick from top candidates
                # To make it faster, we only collect top K
                # The loop above collects ALL with max gain.
                # This is expensive if many ties.
                # For now, just pick one randomly from the list of max gain.
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
                
            selected.append(idx)
            used[idx] = True
            uncovered &= ~masks[idx]
            
        return selected

    # 5. Remove Redundant Blocks
    def remove_redundant(blocks_list):
        if not blocks_list:
            return []
        
        # Sort by mask size? Smaller masks are more likely to be redundant?
        # Or just iterate.
        changed = True
        current_list = blocks_list[:]
        
        while changed:
            changed = False
            for i in range(len(current_list) - 1, -1, -1):
                mask_i = masks[current_list[i]]
                # Check if mask_i is covered by union of others
                union_others = 0
                for j in range(len(current_list)):
                    if j != i:
                        union_others |= masks[current_list[j]]
                        # Early exit if union_others covers mask_i
                        if (union_others & mask_i) == mask_i:
                            break
                
                if (union_others & mask_i) == mask_i:
                    current_list.pop(i)
                    changed = True
                    break # Restart scan after removal
        
        return current_list

    # 6. Branch and Bound (Time Limited)
    start_time = time.time()
    time_limit = 100 # seconds
    
    best_solution = run_greedy()
    best_len = len(best_solution)
    
    # Multi-start greedy with randomization
    for _ in range(20):
        if time.time() - start_time > time_limit - 30:
            break
        sol = run_greedy(randomize=True)
        sol = remove_redundant(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]
            
    # Try to improve best_solution with limited B&B
    # We use a recursive search that tries to find a solution of size < best_len
    # We branch on the uncovered t-subset with the fewest available blocks.
    
    # To make B&B fast, we use a "Lazy" approach or just a deep dive.
    # Given the complexity, a simple B&B might be too slow.
    # We'll rely on the multi-start greedy which is often effective for these sizes.
    
    # One more optimization: Try to swap blocks in best_solution to allow removal.
    # If removing block A leaves gap G, is there a block B not in solution that covers G?
    # If yes, swap A->B. This keeps count same.
    # Then try to remove again.
    
    def try_swap_improve(blocks_list):
        improved = True
        current_list = blocks_list[:]
        
        while improved:
            improved = False
            for i in range(len(current_list)):
                # Try removing block i
                mask_i = masks[current_list[i]]
                union_others = 0
                for j in range(len(current_list)):
                    if j != i:
                        union_others |= masks[current_list[j]]
                        if (union_others & mask_i) == mask_i:
                            break
                
                if (union_others & mask_i) == mask_i:
                    # Redundant
                    current_list.pop(i)
                    improved = True
                    break
                
                # Gap = mask_i & ~union_others
                gap = mask_i & ~union_others
                if gap:
                    # Find a block B not in current_list that covers gap
                    # To be efficient, we can check blocks that cover the "hardest" bit in gap
                    target_bit = gap & -gap
                    target_idx = target_bit.bit_length() - 1
                    
                    candidates = t_to_blocks[target_idx]
                    # Filter candidates: must cover all of gap, and not be in current_list
                    current_set = set(current_list)
                    for b_idx in candidates:
                        if b_idx in current_set:
                            continue
                        if (masks[b_idx] & gap) == gap:
                            # Swap
                            current_list[i] = b_idx
                            improved = True
                            break
                            
                    if improved:
                        break
                        
        return current_list

    best_solution = try_swap_improve(best_solution)
    best_solution = remove_redundant(best_solution)
    
    return [list(k_subsets[i]) for i in best_solution]
