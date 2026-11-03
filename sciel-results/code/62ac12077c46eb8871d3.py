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
    n_k_subsets = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    # We store masks as integers for fast bitwise operations
    masks = [0] * n_k_subsets
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Helper to check if a list of block indices covers all t-subsets
    def is_valid_cover(block_indices):
        cover = 0
        for idx in block_indices:
            cover |= masks[idx]
        return (cover == all_mask)
        
    # Helper to get the union mask of a list of blocks
    def get_union_mask(block_indices):
        cover = 0
        for idx in block_indices:
            cover |= masks[idx]
        return cover

    best_solution = []
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 100 # Leave 20s buffer for safety
    
    # Multi-start Greedy with Local Search
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # --- Phase 1: Greedy Construction ---
        uncovered = all_mask
        selected = []
        used = [False] * n_k_subsets
        
        # Pre-calculate gains? No, dynamic.
        # To speed up greedy, we can keep a list of active candidates.
        # But recalculating gain for 75k items is O(75k). 75k * 300 steps = 22M ops. Feasible.
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            # In later restarts, we can add randomization
            # For now, standard greedy
            for i in range(n_k_subsets):
                if used[i]:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == uncovered.bit_count():
                        break
            
            if best_idx == -1:
                break # Should not happen if problem is coverable
            
            selected.append(best_idx)
            used[best_idx] = True
            uncovered &= ~masks[best_idx]
            
        # --- Phase 2: Local Search (1-to-0 and 2-to-1 swaps) ---
        # We represent the current set of selected block indices
        current_set = set(selected)
        
        # 1-to-0 Swap: Remove block i if it's redundant
        changed = True
        while changed and (time.time() - start_time < time_limit):
            changed = False
            # Try to remove each block
            for i in list(current_set):
                # Check if union of others covers all
                # Optimization: Precompute union of all, then subtract i? 
                # Union is not invertible. We must recompute or maintain counts.
                # Recalculating union of (size-1) sets is expensive if done naively.
                # Let's maintain a count array for how many blocks cover each t-subset.
                pass
        
        # The above 1-to-0 loop with naive union check is O(N^2 * K). 
        # N ~ 300. 300^2 * 300 = 27M. Might be slow in Python.
        # Better: Maintain counts.
        
        # Let's rewrite the local search to be efficient using counts.
        # counts[j] = number of blocks in current_set that cover t-subset j.
        
        current_list = list(current_set)
        # Initialize counts
        counts = [0] * n_t_subsets
        for idx in current_list:
            m = masks[idx]
            # Iterate over set bits in m
            temp = m
            while temp:
                lsb = temp & -temp
                idx_bit = lsb.bit_length() - 1
                counts[idx_bit] += 1
                temp ^= lsb
        
        # 1-to-0 Swap
        changed = True
        while changed and (time.time() - start_time < time_limit):
            changed = False
            for i in range(len(current_list)):
                idx = current_list[i]
                m = masks[idx]
                # Check if any t-subset covered by idx has count == 1
                # If yes, idx is essential.
                # If no, idx is redundant.
                is_redundant = True
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] == 1:
                        is_redundant = False
                        break
                    temp ^= lsb
                
                if is_redundant:
                    # Remove idx
                    # Update counts
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        temp ^= lsb
                    
                    current_list.pop(i)
                    changed = True
                    break # Restart loop as indices/counts changed
        
        # 2-to-1 Swap
        # Try to replace two blocks {i, j} with one block x
        # Condition: Union of (Current \ {i, j}) covers all t-subsets NOT covered by x?
        # No, (Current \ {i, j}) ∪ {x} must cover ALL.
        # This is equivalent to: All t-subsets that were covered ONLY by i or ONLY by j or BOTH
        # must be covered by x OR still covered by others.
        
        # Efficient check:
        # Let U = Union(Current).
        # We need (Union(Current \ {i, j}) | masks[x]) == all_mask.
        
        # We can precompute union of current_list?
        # Union changes every step.
        
        # Heuristic 2-to-1:
        # For each pair (i, j) in current_set, identify the "gap" created by removing them.
        # Gap = all_mask & ~(Union(Current \ {i, j}))
        # We need a block x such that Gap ⊆ masks[x].
        
        # This is expensive to compute for all pairs.
        # Approximation: Only try pairs where i and j overlap significantly?
        
        # Given time constraints, 1-to-0 is the most robust and fast improvement.
        # 2-to-1 is harder.
        
        # Let's add a simple 2-to-1 check for a limited number of iterations.
        # We'll just do 1-to-0 effectively.
        
        if len(current_list) < best_size:
            best_size = len(current_list)
            best_solution = current_list[:]
            
        # If we have time, try a few random 2-to-1 swaps?
        # Let's stick to 1-to-0 for reliability and speed.
        
    # Fallback if no solution found (should not happen)
    if not best_solution:
        # Greedy without local search
        uncovered = all_mask
        selected = []
        used = [False] * n_k_subsets
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(n_k_subsets):
                if used[i]:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1: break
            selected.append(best_idx)
            used[best_idx] = True
            uncovered &= ~masks[best_idx]
        best_solution = selected
        
    return [list(k_subsets[i]) for i in best_solution]
