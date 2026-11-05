import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    # 90s limit. Use 85s for search, 5s for safety/verification.
    deadline = start_time + 85.0
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Precompute conflict masks
    # conflict_mask[i] is an integer where bit j is 1 if candidates[i] and candidates[j] conflict
    # (i.e., lcm(candidates[i], candidates[j]) <= N)
    conflict_mask = [0] * n
    for i in range(n):
        a = candidates[i]
        mask = 0
        for j in range(i + 1, n):
            b = candidates[j]
            if math.lcm(a, b) <= N:
                mask |= (1 << j)
                conflict_mask[j] |= (1 << i)
        conflict_mask[i] = mask
        
    # Precompute weights (reciprocals)
    weights = [1.0 / candidates[i] for i in range(n)]
    
    # Helper to check if a set (represented by a bitmask) is valid
    # We maintain the invariant that the current set is valid.
    
    # 1. Greedy Initialization
    # Sort candidates by weight descending (smallest number first)
    # To break ties or add variation, we can shuffle slightly or just use standard order.
    # Standard order: 2, 3, 4...
    # Greedy: Add if compatible with all currently selected.
    
    def greedy_init(shuffle_factor=0.0):
        """
        Builds a set using a greedy approach.
        shuffle_factor > 0 introduces randomness in the order of consideration.
        """
        indices = list(range(n))
        if shuffle_factor > 0:
            # Partial shuffle: mix up the lower indices more? Or full shuffle?
            # Let's do a weighted random permutation where lower indices (higher weight) 
            # have higher probability of being picked earlier, but with noise.
            # Simpler: Just shuffle and pick. But pure shuffle might miss the obvious good start.
            # Let's stick to a deterministic greedy first, then random restarts.
            random.shuffle(indices)
        else:
            # Sort by weight descending (which is index ascending since weights = 1/(i+2))
            indices = sorted(indices, key=lambda i: -weights[i])
            
        selected_mask = 0
        current_sum = 0.0
        
        for i in indices:
            # Check if i is compatible with all selected
            # i is compatible with selected set S if (conflict_mask[i] & selected_mask) == 0
            if (conflict_mask[i] & selected_mask) == 0:
                selected_mask |= (1 << i)
                current_sum += weights[i]
                
        return selected_mask, current_sum

    best_mask, best_sum = greedy_init(0.0)
    
    # 2. Local Search (Hill Climbing)
    # Moves:
    # a) Add a candidate
    # b) Remove a candidate
    # c) Swap: Remove one, Add another
    
    def get_indices(mask):
        """Return list of indices set in mask."""
        return [i for i in range(n) if (mask >> i) & 1]
    
    def is_valid_addition(mask, i):
        """Check if i can be added to mask."""
        return (conflict_mask[i] & mask) == 0 and not ((mask >> i) & 1)
        
    def local_search(mask, current_sum, time_limit):
        """
        Improves the given mask using local search.
        Returns improved mask and sum.
        """
        improved = True
        last_improvement_time = time.time()
        
        while improved and time.time() < time_limit:
            improved = False
            # Try all possible add moves
            # To speed up, we can iterate through bits not in mask
            not_in_mask = ((1 << n) - 1) ^ mask
            
            # Try Adds
            best_gain = 0.0
            best_add_idx = -1
            
            # Iterate through potential additions
            # Optimization: Only consider indices that are compatible.
            # We can filter the 'not_in_mask' by checking conflicts.
            # But checking 255 items is fast.
            
            m = not_in_mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                if (conflict_mask[i] & mask) == 0:
                    gain = weights[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_add_idx = i
                m ^= lsb
                
            if best_add_idx != -1:
                mask |= (1 << best_add_idx)
                current_sum += best_gain
                improved = True
                last_improvement_time = time.time()
                continue # Restart search loop as mask changed significantly
            
            # Try Swaps (Remove one, Add another)
            # This is O(n^2). With n=255, that's ~65k checks. Very fast in Python if optimized?
            # Might be slow in pure Python loops. Let's limit swap attempts or do it smartly.
            # Actually, if no single add is possible, a swap might be.
            
            # To avoid O(n^2) overhead in Python every iteration, we can try a limited number of swaps
            # or only check swaps involving the lowest weight elements in the current set.
            
            # Strategy: Try to remove the element with the smallest weight (largest index) 
            # and see if we can add a better one.
            # Or just iterate all removals and check best addition.
            
            selected_indices = get_indices(mask)
            if not selected_indices:
                break
                
            # Sort selected by weight ascending (weakest first) to try removing them
            # But we need to check if removing one allows adding a much better one.
            
            best_swap_gain = 0.0
            best_remove_idx = -1
            best_add_idx_swap = -1
            
            # Limit swap search to first 20 weakest elements to keep it fast?
            # Or just do full O(n^2) if n is small enough? 
            # 255^2 = 65025. In Python, this loop takes ~0.01-0.05s. 
            # If we do this 1000 times, it's 10-50s. Too slow.
            
            # Optimization: 
            # For each candidate removal r in selected:
            #   new_mask = mask without r
            #   best_add = max weight i such that i not in new_mask and compatible with new_mask
            #   gain = weight[i] - weight[r]
            
            # We can precompute the "best available addition" for any mask? No, mask changes.
            
            # Let's try a heuristic: 
            # Only consider removing the bottom 10% of elements by weight.
            # And only try adding from the top 50% of elements by weight.
            
            # Simpler approach for this step:
            # Iterate through a random subset of swaps to avoid getting stuck in slow O(n^2) loops repeatedly.
            # Or, just do the O(n^2) but break early if time is tight.
            
            # Let's try the O(n^2) but only if we haven't improved in a while?
            # Actually, let's just do the Add move. If stuck, try a random restart or limited swap.
            
            # For now, let's implement a limited swap search:
            # Check swaps for the 10 weakest elements in the current set.
            weakest_selected = sorted(selected_indices, key=lambda i: weights[i])[:10]
            
            # Candidate additions: all elements not in mask, sorted by weight descending, top 50
            potential_adds = [i for i in range(n) if not ((mask >> i) & 1)]
            potential_adds.sort(key=lambda i: -weights[i])
            top_adds = potential_adds[:50]
            
            for r in weakest_selected:
                mask_without_r = mask ^ (1 << r)
                # Find best add compatible with mask_without_r
                for a in top_adds:
                    if a == r: continue
                    if (conflict_mask[a] & mask_without_r) == 0:
                        gain = weights[a] - weights[r]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_remove_idx = r
                            best_add_idx_swap = a
                            
            if best_remove_idx != -1:
                mask = (mask ^ (1 << best_remove_idx)) | (1 << best_add_idx_swap)
                current_sum += best_swap_gain
                improved = True
                last_improvement_time = time.time()
                continue

            # If no improvement found in this pass, try a random perturbation (Simulated Annealing style)
            # to escape local minima.
            # Accept a worse move with some probability? 
            # Or just do a random restart if stuck for too long.
            
            # Let's check time. If we are stuck, do a random restart.
            if time.time() - last_improvement_time > 2.0:
                break
                
        return mask, current_sum

    # Main Search Loop: Multiple Restarts
    # 1. Deterministic Greedy
    # 2. Local Search on it
    # 3. Randomized Greedies + Local Search
    
    best_mask, best_sum = local_search(best_mask, best_sum, deadline - 5.0)
    
    # Random Restarts
    # We have ~80s left.
    # Each restart: Greedy (fast) + Local Search (variable).
    # Let's budget 0.5s per restart? That allows ~160 restarts.
    # Or just run until deadline.
    
    restart_count = 0
    while time.time() < deadline - 2.0:
        restart_count += 1
        # Randomized Greedy
        # Use a random seed for reproducibility if needed, but here we just want variety.
        # We can bias the shuffle: pick next element from a random subset of the top remaining?
        # Simpler: Full random shuffle of indices.
        
        # To make it more effective, let's use a "noisy" greedy:
        # At each step, pick the best compatible element from a random sample of size K.
        # Or just full shuffle.
        
        mask, curr_sum = greedy_init(1.0) # Full shuffle
        
        # Run local search with a tighter time limit per restart?
        # If we run local search until deadline, we only get one restart.
        # We need to limit local search time per restart.
        
        # Estimate time for local search. It's hard.
        # Let's allocate 1s per restart for local search.
        search_deadline = time.time() + 1.0
        if search_deadline > deadline - 2.0:
            search_deadline = deadline - 2.0
            
        mask, curr_sum = local_search(mask, curr_sum, search_deadline)
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_mask = mask
            
    # Verification
    # Ensure the best_mask is valid
    indices = get_indices(best_mask)
    valid = True
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            if math.lcm(candidates[indices[i]], candidates[indices[j]]) <= N:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set if something went wrong (shouldn't happen)
        # Greedy deterministic is always valid.
        best_mask, best_sum = greedy_init(0.0)
        indices = get_indices(best_mask)
        
    if not indices:
        return [2]
        
    return [candidates[i] for i in indices]

