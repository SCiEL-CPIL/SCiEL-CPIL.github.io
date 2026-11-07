import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    def is_valid_sums(sums):
        """
        Check if no two distinct sums in 'sums' divide each other.
        sums is a set of integers.
        """
        if not sums:
            return True
        
        # Sort to make it easier to check multiples
        sorted_sums = sorted(sums)
        n = len(sorted_sums)
        
        # For each sum s, check if any multiple 2s, 3s... is in the set.
        # Since we sorted, if s divides t, then t >= 2s.
        # We can just iterate through the set.
        # Optimization: Use a set for O(1) lookups.
        sums_set = sums
        
        for s in sorted_sums:
            # Check multiples
            multiple = 2 * s
            while multiple <= sorted_sums[-1]:
                if multiple in sums_set:
                    return False
                multiple += s
        return True

    def get_subset_sums(elements):
        """
        Compute all non-empty subset sums.
        """
        sums = set()
        n = len(elements)
        # Bitmask iteration
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def try_add_element(current_elements, current_sums, candidate):
        """
        Try to add 'candidate' to 'current_elements'.
        Returns (new_elements, new_sums) if valid, else (None, None).
        """
        # New sums are: old_sums U {s + candidate for s in old_sums} U {candidate}
        new_sums = set(current_sums)
        
        # Add candidate
        if candidate in new_sums:
            return None, None
        new_sums.add(candidate)
        
        # Add shifted sums
        for s in current_sums:
            ns = s + candidate
            if ns in new_sums:
                # Collision: two subsets have same sum.
                # This implies invalidity because if sum is s, and another subset has sum s,
                # they are distinct subsets.
                # Wait, the problem says "Equal sum values are counted once. Two different subsets with the same sum still make the set invalid".
                # So if ns is already in new_sums (which contains old sums and candidate), it's a collision.
                return None, None
            new_sums.add(ns)
        
        # Check divisibility
        if not is_valid_sums(new_sums):
            return None, None
            
        new_elements = current_elements | {candidate}
        return new_elements, new_sums

    def greedy_construct(candidates):
        current_elements = set()
        current_sums = set()
        
        for c in candidates:
            if time.time() > deadline:
                break
            new_elems, new_sums = try_add_element(current_elements, current_sums, c)
            if new_elems is not None:
                current_elements = new_elems
                current_sums = new_sums
                
        return current_elements, current_sums

    def local_search(elements, sums, time_limit):
        """
        Try to improve the set by swapping elements.
        """
        current_elements = set(elements)
        current_sums = sums
        best_size = len(current_elements)
        
        # Pre-compute list of candidates not in current set
        all_candidates = set(range(1, 257))
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add a new element
            # Sample some random candidates not in current set
            pool = list(all_candidates - current_elements)
            random.shuffle(pool)
            
            for c in pool[:50]: # Try 50 random candidates
                new_elems, new_sums = try_add_element(current_elements, current_sums, c)
                if new_elems is not None:
                    current_elements = new_elems
                    current_sums = new_sums
                    best_size = len(current_elements)
                    improved = True
                    break # Restart adding loop with new state
            
            if improved:
                continue
                
            # 2. Try to swap
            # Pick a random element from current set to remove
            if not current_elements:
                break
                
            to_remove = random.choice(list(current_elements))
            
            # Remove it
            reduced_elements = current_elements - {to_remove}
            
            # Recompute sums for reduced set (expensive but necessary for accuracy if we don't track structure)
            # Optimization: We can try to add candidates to the reduced set.
            # If we find a candidate that fits, we have the same size but different composition.
            # This might allow adding a 3rd element later?
            
            # Actually, a better swap strategy:
            # Try to replace 'to_remove' with 'c' from pool.
            # Check if (current_elements - {to_remove} U {c}) is valid.
            
            pool = list(all_candidates - current_elements)
            random.shuffle(pool)
            
            found_swap = False
            for c in pool[:20]:
                if c == to_remove:
                    continue
                    
                new_elements = (current_elements - {to_remove}) | {c}
                
                # Check validity of new_elements
                # We need sums of new_elements.
                # This requires recomputing sums.
                # To speed up, we can check if the size is still the same.
                # If we just want to improve size, swaps don't help directly unless they enable a future add.
                # However, if we are stuck at size K, we might need to change composition to find a K+1.
                
                # Heuristic: Only do expensive swap checks if we are close to deadline or stuck for a while.
                # For now, let's just try to add. If we can't add, we restart greedy.
                
                # Let's implement a simple "recompute and check" for swaps, but limit it.
                # If len(new_elements) == len(current_elements):
                #     new_sums = get_subset_sums(list(new_elements))
                #     if is_valid_sums(new_sums):
                #         # Accept swap to explore
                #         current_elements = new_elements
                #         current_sums = new_sums
                #         found_swap = True
                #         break
                #         # Note: This doesn't increase size, but might change the "shape" of sums.
                
            if found_swap:
                continue
                
            # If we couldn't add or swap, break and let outer loop restart
            break
            
        return current_elements, current_sums

    # Main Search
    best_set = set()
    best_sums = set()
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    candidates = list(range(1, 257))
    
    # Ordering 1: Descending (Large numbers first)
    if time.time() < deadline - 5.0:
        res_elems, res_sums = greedy_construct(sorted(candidates, reverse=True))
        if len(res_elems) > best_size:
            best_set, best_sums, best_size = res_elems, res_sums, len(res_elems)
            
    # Ordering 2: Ascending
    if time.time() < deadline - 5.0:
        res_elems, res_sums = greedy_construct(sorted(candidates))
        if len(res_elems) > best_size:
            best_set, best_sums, best_size = res_elems, res_sums, len(res_elems)
            
    # Ordering 3: Random restarts
    for i in range(200):
        if time.time() > deadline - 10.0:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        res_elems, res_sums = greedy_construct(shuffled)
        
        if len(res_elems) > best_size:
            best_set, best_sums, best_size = res_elems, res_sums, len(res_elems)
            
        # Local search on the result
        if len(res_elems) > 5: # Only if worth it
            ls_elems, ls_sums = local_search(res_elems, res_sums, deadline - 5.0)
            if len(ls_elems) > best_size:
                best_set, best_sums, best_size = ls_elems, ls_sums, len(ls_elems)

    # Final Verification
    if best_set:
        elems = sorted(best_set)
        sums = get_subset_sums(elems)
        if is_valid_sums(sums):
            return [int(x) for x in elems]
            
    # Fallback
    return [3]
