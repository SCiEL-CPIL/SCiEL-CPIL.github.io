import time
import random

def construct():
    deadline = time.time() + 110  # 10s margin
    
    N = 284
    fixed = [1, 2, 5, 11]
    
    def calc_mask(arr):
        mask = 0
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                s = arr[i] + arr[j]
                mask |= (1 << (s - 2))
        return mask

    init_mask = calc_mask(fixed)
    candidates = [x for x in range(1, N + 1) if x not in fixed]
    candidates.sort()
    
    best_set = fixed[:]
    best_size = len(fixed)
    
    # 1. Randomized Greedy
    # Try to find a large set quickly
    for _ in range(200):
        if time.time() > deadline - 50:
            break
        cand = candidates[:]
        random.shuffle(cand)
        curr = fixed[:]
        curr_mask = init_mask
        for x in cand:
            new_bits = 0
            for y in curr:
                new_bits |= (1 << (x + y - 2))
            new_bits |= (1 << (2 * x - 2))
            
            if (new_bits & curr_mask) == 0:
                curr.append(x)
                curr_mask |= new_bits
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
            
    # 2. Backtracking
    # We search for size > best_size
    
    # Helper to check if we can beat best_size using a greedy pass on remaining candidates
    def greedy_bound(curr_set, curr_mask, start_idx):
        temp_mask = curr_mask
        count = 0
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            new_bits = 0
            for y in curr_set:
                new_bits |= (1 << (x + y - 2))
            new_bits |= (1 << (2 * x - 2))
            
            if (new_bits & temp_mask) == 0:
                # Add it to the simulation
                # We don't actually need to update curr_set for the mask calculation of future elements?
                # Wait, new_bits for future elements depends on the set.
                # So we must update a temporary set or just track the mask?
                # The mask is not enough to calculate new_bits for the next x, 
                # because new_bits depends on the elements y in the set.
                # So we need to maintain a temporary set.
                # This makes the bound O(K * N) where K is size.
                # This might be too slow.
                
                # Alternative: Just use the simple count bound?
                # Or a weaker bound: count how many candidates are individually compatible with curr_set.
                # This is O(N) per node.
                
                pass
        return 0 # Placeholder

    # Let's use a simpler bound: 
    # Max possible size = len(curr) + len(remaining)
    # And we rely on the search order.
    
    def search(curr_set, curr_mask, cand_idx):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        # Simple Pruning
        if len(curr_set) + (len(candidates) - cand_idx) <= best_size:
            return
            
        if cand_idx == len(candidates):
            if len(curr_set) > best_size:
                best_size = len(curr_set)
                best_set = curr_set[:]
            return

        x = candidates[cand_idx]
        
        # Calculate new bits
        new_bits = 0
        for y in curr_set:
            new_bits |= (1 << (x + y - 2))
        new_bits |= (1 << (2 * x - 2))
        
        # Option 1: Add x
        if (new_bits & curr_mask) == 0:
            curr_set.append(x)
            search(curr_set, curr_mask | new_bits, cand_idx + 1)
            curr_set.pop()
            
        # Option 2: Skip x
        search(curr_set, curr_mask, cand_idx + 1)

    # Start search
    # To improve efficiency, we can try to search for a specific target size first?
    # No, standard B&B.
    
    # If the search is too slow, we might not finish.
    # But the random greedy should have found a good best_size.
    
    search(fixed[:], init_mask, 0)
    
    return best_set
