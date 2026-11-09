"""Rewritten program: maximize valid subset of [1,256] where (a*b)%(a+b)!=0 for all distinct a,b."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s kill
    
    # Precompute conflict bitmasks
    # conflict_mask[i] is a bitmask where bit j is set if (i+1) and (j+1) conflict
    n = 256
    conflict_mask = [0] * n
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def greedy_build(order):
        """Greedily add elements in the given order if they don't conflict with existing set."""
        selected_set_mask = 0
        selected_list = []
        for idx in order:
            # Check if idx conflicts with any element in selected_set_mask
            # If (conflict_mask[idx] & selected_set_mask) == 0, no conflict
            if (conflict_mask[idx] & selected_set_mask) == 0:
                selected_set_mask |= (1 << idx)
                selected_list.append(idx)
        return selected_list, selected_set_mask

    def local_search(selected_list, selected_set_mask, max_iters=200):
        """
        Improve the set by adding compatible elements and swapping.
        Uses bitmasks for fast conflict checks.
        """
        current_size = len(selected_list)
        # List of elements NOT in the set, for faster iteration in some cases
        # But we can just iterate all and check membership via bitmask
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            
            # 1. Try to add any element that doesn't conflict
            # We can find all addable elements at once:
            # An element i is addable if (conflict_mask[i] & selected_set_mask) == 0
            # Let's just iterate through all indices and check
            for idx in range(n):
                if (selected_set_mask >> idx) & 1:
                    continue # Already in set
                if (conflict_mask[idx] & selected_set_mask) == 0:
                    # Can add
                    selected_set_mask |= (1 << idx)
                    selected_list.append(idx)
                    current_size += 1
                    improved = True
                    break # Restart loop to check for more additions
            
            if improved:
                continue
            
            # 2. Try swaps: remove one, add another
            # To keep it fast, we limit the number of removal candidates we check per iteration
            # Randomly pick a few elements from the set to try removing
            if len(selected_list) > 0:
                # Pick up to 5 random elements to try removing
                removal_candidates = random.sample(selected_list, min(5, len(selected_list)))
                for rem_idx in removal_candidates:
                    new_set_mask = selected_set_mask & ~(1 << rem_idx)
                    
                    # Find if any element outside the new set can be added
                    # Ideally we want to add an element that allows us to keep the size or increase it
                    # Since we remove 1, we need to add at least 1 to maintain size, or 2 to improve.
                    # Simple strategy: try to find ANY addable element in the new set context.
                    # If we find one, we do the swap (size stays same). 
                    # If we find two? We can only add one per step usually, but let's just add one.
                    # Actually, if we remove 1 and add 1, size is same. 
                    # To improve size, we would need to remove 1 and add 2, which is complex.
                    # Let's stick to 1-for-1 swaps to escape local minima where no single add is possible.
                    
                    found_add = -1
                    for idx in range(n):
                        if (new_set_mask >> idx) & 1:
                            continue
                        if (conflict_mask[idx] & new_set_mask) == 0:
                            found_add = idx
                            break
                    
                    if found_add != -1:
                        # Perform swap
                        selected_set_mask = new_set_mask | (1 << found_add)
                        # Update list: remove rem_idx, add found_add
                        selected_list.remove(rem_idx)
                        selected_list.append(found_add)
                        improved = True
                        break
            
            if not improved:
                break
        
        return selected_list, selected_set_mask, current_size

    best_result_mask = 0
    best_result_list = []
    best_size = 0
    
    # Strategy 1: Greedy with specific orderings
    orderings = [
        list(range(n)), # Ascending
        list(range(n-1, -1, -1)), # Descending
        [i for i in range(n) if (i+1) % 2 == 1] + [i for i in range(n) if (i+1) % 2 == 0], # Odds first
        [i for i in range(n) if (i+1) % 2 == 0] + [i for i in range(n) if (i+1) % 2 == 1], # Evens first
    ]
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        sel_list, sel_mask = greedy_build(order)
        sel_list, sel_mask, sz = local_search(sel_list, sel_mask, max_iters=100)
        if sz > best_size:
            best_size = sz
            best_result_list = sel_list
            best_result_mask = sel_mask
    
    # Strategy 2: Randomized restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        order = list(range(n))
        random.shuffle(order)
        sel_list, sel_mask = greedy_build(order)
        sel_list, sel_mask, sz = local_search(sel_list, sel_mask, max_iters=50)
        if sz > best_size:
            best_size = sz
            best_result_list = sel_list
            best_result_mask = sel_mask
        
        # If we've done many restarts, reduce iterations to fit more in
        if restarts > 100:
             # Maybe stop if not improving? For now, just keep going until time limit.
             pass

    # Final polish: Try to add any remaining elements to the best set
    if best_result_list:
        sel_list, sel_mask = local_search(best_result_list, best_result_mask, max_iters=50)
        if len(sel_list) > best_size:
            best_result_list = sel_list
            best_result_mask = sel_mask
            best_size = len(sel_list)
        
        # One last manual pass to ensure no addable elements were missed
        changed = True
        while changed:
            changed = False
            for idx in range(n):
                if (best_result_mask >> idx) & 1:
                    continue
                if (conflict_mask[idx] & best_result_mask) == 0:
                    best_result_mask |= (1 << idx)
                    best_result_list.append(idx)
                    best_size += 1
                    changed = True

    if not best_result_list:
        # Fallback: Just pick 1
        return [1]

    # Convert indices back to values
    result = [i + 1 for i in best_result_list]
    result.sort()
    
    return result
