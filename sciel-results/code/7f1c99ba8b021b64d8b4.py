import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.monotonic() + 620.0  # Safety margin for 630s limit

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        f = (x * x) % m
        if f != 0:
            forbidden.add(f)
    forbidden.discard(0)

    # 2. Precompute conflict masks
    # conflict_mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def is_feasible(subset_mask, candidate):
        """Check if candidate can be added to subset_mask."""
        if subset_mask & (1 << candidate):
            return False
        return (conflict_mask[candidate] & subset_mask) == 0

    def popcount(mask):
        return bin(mask).count('1')

    def mask_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << x)
        return mask

    # 3. Greedy Construction
    def greedy_construct(order):
        subset = 0
        for cand in order:
            if is_feasible(subset, cand):
                subset |= (1 << cand)
        return subset

    # 4. Local Search
    def local_search(initial_mask, rng, max_iters=1000):
        current = initial_mask
        cur_size = popcount(current)
        best = current
        best_size = cur_size
        
        all_elements = list(range(m))
        
        for _ in range(max_iters):
            if time.monotonic() > deadline:
                break
            
            improved = False
            
            # Phase A: Try to add any element
            # Iterate through all elements not in current
            # Optimization: Precompute list of not-in elements? 
            # For N=205, just checking bits is fast.
            
            # Try adding
            for cand in all_elements:
                if not (current & (1 << cand)):
                    if is_feasible(current, cand):
                        current |= (1 << cand)
                        cur_size += 1
                        improved = True
                        if cur_size > best_size:
                            best = current
                            best_size = cur_size
                        # Restart add phase to find more adds
                        break
            
            if improved:
                continue
            
            # Phase B: Try to swap
            # For each element in current, try to remove it and add another
            # To speed up, we can just iterate through all pairs (remove, add)
            # But that's 205*205 = 42k checks. Still fast.
            
            # Let's do a randomized swap search to avoid getting stuck in a specific local optimum
            # Or systematic swap search.
            
            # Systematic Swap:
            in_current = [x for x in all_elements if (current & (1 << x))]
            not_in_current = [x for x in all_elements if not (current & (1 << x))]
            
            # To make it faster, we can randomize the order of in_current and not_in_current
            rng.shuffle(in_current)
            rng.shuffle(not_in_current)
            
            found_swap = False
            for rem in in_current:
                if time.monotonic() > deadline:
                    break
                temp_mask = current & ~(1 << rem)
                
                for add in not_in_current:
                    if add == rem:
                        continue
                    if is_feasible(temp_mask, add):
                        # Valid swap
                        current = temp_mask | (1 << add)
                        # Size remains same, but structure changes
                        # We don't update best_size here as size didn't change
                        # But we update current state
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if found_swap:
                continue
            
            # Phase C: No add, no swap found. Local optimum.
            # Apply a kick: remove k random elements, add k random feasible ones?
            # Or just restart from a slightly perturbed state.
            # Let's try removing 2 random elements and adding 2 random feasible ones.
            
            if len(in_current) >= 2:
                # Remove 2 random
                rem1, rem2 = rng.sample(in_current, 2)
                temp_mask = current & ~(1 << rem1) & ~(1 << rem2)
                
                # Try to add 2 random feasible elements
                # We need to find elements that are feasible with temp_mask
                # And feasible with each other.
                
                # Simple approach: pick random candidates from not_in_current
                # Check if they fit in temp_mask
                # If we find one, add it. Then find another.
                
                added_count = 0
                # We want to potentially increase size, so we try to add 2.
                # If we can only add 1, size drops by 1. 
                # If we add 2, size stays same.
                # If we add 3, size increases by 1.
                
                # Let's just try to find any 2 elements that can be added to temp_mask.
                candidates = [x for x in not_in_current if x != rem1 and x != rem2]
                rng.shuffle(candidates)
                
                found_adds = []
                for cand in candidates:
                    if len(found_adds) == 2:
                        break
                    if is_feasible(temp_mask, cand):
                        # Check if it conflicts with previously found adds in this kick
                        conflict_with_prev = False
                        for prev in found_adds:
                            if (conflict_mask[cand] >> prev) & 1:
                                conflict_with_prev = True
                                break
                        if not conflict_with_prev:
                            found_adds.append(cand)
                
                # Apply kick if we found at least 1 add?
                # If we remove 2 and add 0, size -2. Bad.
                # If we remove 2 and add 1, size -1. Bad.
                # If we remove 2 and add 2, size 0. Neutral.
                # If we remove 2 and add 3, size +1. Good.
                
                # So we only apply the kick if we found >= 2 adds.
                if len(found_adds) >= 2:
                    new_mask = temp_mask
                    for a in found_adds:
                        new_mask |= (1 << a)
                    current = new_mask
                    cur_size = popcount(current)
                    if cur_size > best_size:
                        best = current
                        best_size = cur_size
            else:
                # If set is small, just break or restart
                break

        return best, best_size

    # 5. Main Search
    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Starts
    # Try different orderings
    orderings = []
    # Random
    for _ in range(50):
        order = list(range(m))
        rng.shuffle(order)
        orderings.append(order)
    # Degree based
    degrees = [bin(conflict_mask[i]).count('1') for i in range(m)]
    order_low_deg = sorted(range(m), key=lambda x: degrees[x])
    order_high_deg = sorted(range(m), key=lambda x: -degrees[x])
    orderings.append(order_low_deg)
    orderings.append(order_high_deg)
    
    initial_solutions = []
    for order in orderings:
        sol = greedy_construct(order)
        initial_solutions.append(sol)
        if popcount(sol) > best_size:
            best_mask = sol
            best_size = popcount(sol)
            
    # Phase 2: Local Search on Best and Random Restarts
    # We will run local search on the current best, and periodically restart from random greedy solutions
    
    # Keep a list of best solutions found
    best_solutions = [best_mask]
    
    while time.monotonic() < deadline - 10:
        # Pick a starting point
        # 80% chance start from best, 20% from random greedy
        if rng.random() < 0.8 and best_solutions:
            start_mask = rng.choice(best_solutions)
        else:
            order = list(range(m))
            rng.shuffle(order)
            start_mask = greedy_construct(order)
            
        # Run local search
        improved_mask, improved_size = local_search(start_mask, rng, max_iters=200)
        
        if improved_size > best_size:
            best_mask = improved_mask
            best_size = improved_size
            best_solutions.append(improved_mask)
            # Keep only top 5 best solutions to avoid memory bloat and redundant searches
            if len(best_solutions) > 5:
                best_solutions.sort(key=lambda x: -popcount(x))
                best_solutions = best_solutions[:5]
        
        # Even if not improved, we might want to add it if it's distinct and decent?
        # No, stick to the top performers.
        
    # Final check
    if best_mask == 0:
        best_mask = 1 # element 0

    result_set = mask_to_list(best_mask)
    result = np.array(sorted(result_set), dtype=np.int64)
    return result
