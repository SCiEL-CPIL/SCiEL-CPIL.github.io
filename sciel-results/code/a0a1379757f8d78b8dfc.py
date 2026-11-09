import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Bitmask for sums 0..200
    mask = (1 << 201) - 1
    
    def greedy(order):
        """
        Greedy construction: iterate through numbers in 'order',
        adding a number if it doesn't create a subset sum of 200.
        Returns (set_of_selected, reachable_bitmask).
        """
        selected = set()
        reachable = 1  # Bit 0 is set
        
        for x in order:
            if x in selected:
                continue
            # Check if adding x creates sum 200
            # This happens if (200 - x) is currently reachable
            if (reachable >> (200 - x)) & 1:
                continue
            
            selected.add(x)
            # Update reachable sums: new_sums = (reachable << x) & mask
            # We can add x to any existing sum s if s+x <= 200
            # The bit at position s+x is set if bit s was set.
            new_bits = (reachable << x) & mask
            reachable |= new_bits
            
        return selected, reachable

    def local_search(selected, reachable):
        """
        Try to add more elements to the selected set.
        Returns updated (selected, reachable).
        """
        current_selected = set(selected)
        current_reachable = reachable
        improved = True
        
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try to add any element from 1..100 not in current_selected
            # Sort by some heuristic? Or random?
            # Let's try all available elements.
            available = [x for x in range(1, 101) if x not in current_selected]
            # Heuristic: try smaller elements first? Or larger?
            # Larger elements are "safer" to add because they block fewer smaller sums?
            # Actually, adding a large number x requires 200-x not reachable.
            # If 200-x is small, it's more likely to be reachable.
            # So large x (small 200-x) is harder to add.
            # Small x (large 200-x) is easier to add if 200-x is not reachable.
            # However, small x creates many new small sums, potentially blocking future large adds.
            # Greedy from large to small is standard.
            # For local search, let's try adding the "easiest" one first?
            # Or just iterate 1..100.
            
            # Let's iterate 1..100 to see if we can squeeze in small numbers.
            for x in available:
                if time.time() > deadline:
                    break
                if (current_reachable >> (200 - x)) & 1:
                    continue
                
                # Safe to add
                current_selected.add(x)
                new_bits = (current_reachable << x) & mask
                current_reachable |= new_bits
                improved = True
                break  # Restart scan to allow for cascading additions
                
        return current_selected, current_reachable

    best_set = set()
    best_size = 0
    best_reachable = 0

    # 1. Deterministic Greedy Attempts
    orders = [
        list(range(100, 0, -1)),  # Descending
        list(range(1, 101)),      # Ascending
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s, r = greedy(order)
        s, r = local_search(s, r)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_reachable = r

    # 2. Randomized Restart Greedy
    # Use bitset for speed.
    max_iterations = 20000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        order = list(range(1, 101))
        random.shuffle(order)
        
        s, r = greedy(order)
        s, r = local_search(s, r)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_reachable = r
            
        # Early stop if we hit a theoretical upper bound?
        # Max size is 100. But we know it's less.
        # If we get 50+, that's good.
        
    # 3. Local Search on Best Set (Swap-based)
    # Try to swap elements to free up space for more additions.
    # This is more complex. Given time, the add-only local search might be sufficient
    # if the initial greedy is good.
    # However, a swap could allow adding 2 elements where previously only 1 was possible.
    
    # Let's try a simple swap heuristic if we have time.
    if time.time() < deadline - 5:
        current_set = set(best_set)
        current_reachable = best_reachable
        
        # Identify "blockers": elements x in current_set such that
        # removing x allows adding some y not in current_set.
        
        # Simpler: try removing each element one by one, see if we can add more.
        # This is O(N^2) roughly with bitset checks.
        
        # For each element x in current_set:
        #   Remove x -> new_reachable
        #   Try to add as many elements as possible via local_search
        #   If size > current_size, keep it.
        
        # This is expensive if local_search is deep.
        # Let's do a shallow version: just check if *one* new element can be added.
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
            
            candidates_to_remove = list(current_set)
            random.shuffle(candidates_to_remove)
            
            for x in candidates_to_remove:
                if time.time() > deadline - 2:
                    break
                
                # Temporarily remove x
                # We need to recompute reachable without x.
                # This requires O(N) bitset operations.
                # Since N <= 100, this is fast.
                
                # Rebuild reachable from scratch without x?
                # Or store parent pointers? Too complex.
                # Rebuild is ~100 shifts/ORs. Very fast.
                
                temp_set = current_set - {x}
                temp_reachable = 1
                for y in temp_set:
                    new_bits = (temp_reachable << y) & mask
                    temp_reachable |= new_bits
                
                # Try to add one element
                added = False
                for y in range(1, 101):
                    if y in temp_set:
                        continue
                    if (temp_reachable >> (200 - y)) & 1:
                        continue
                    
                    # Can add y
                    new_set = temp_set | {y}
                    new_reachable = temp_reachable | ((temp_reachable << y) & mask)
                    
                    # Check if we can add MORE than just y
                    # Run local_search on this new state
                    final_set, final_reach = local_search(new_set, new_reachable)
                    
                    if len(final_set) > len(current_set):
                        current_set = final_set
                        current_reachable = final_reach
                        improved = True
                        break # Restart removal loop with new best
                    elif len(final_set) == len(current_set):
                        # Same size, but maybe different composition allows future swaps?
                        # Stick to size maximization for now.
                        pass
                    break # Only try one add per removal to keep it fast? 
                         # Actually, if we added y, we might be able to add z too.
                         # local_search handles that.
                         # If no y was added, try next x.
                         # Wait, the 'break' above is inside the y-loop.
                         # If we found a y and updated, we break y-loop and restart x-loop.
                         # If we found a y but didn't improve size (unlikely if local_search works), 
                         # we break y-loop? No, we should check other y's?
                         # If adding y doesn't increase size, adding z might?
                         # But if adding y is valid, local_search will try to add more.
                         # If local_search returns same size, it means no further adds were possible.
                         # So trying another y for the same x is unlikely to help if local_search is exhaustive for adds.
                         # So breaking y-loop after first successful add (even if size didn't grow? 
                         # Actually if size didn't grow, it's not a valid improvement. 
                         # But maybe a different y leads to a larger set?
                         # Let's not break y-loop unless we improved.
                         # But checking all y for each x is O(N^2). 
                         # Let's just check if *any* y can be added. 
                         # If yes, do it. If it leads to improvement, keep.
                         # If not, try next x.
                # End y loop
                
                # If we didn't improve, continue to next x
                # If we did improve, 'improved' is True and we break x loop (via while restart)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            best_size = len(best_set)
            best_reachable = current_reachable

    return list(best_set)
