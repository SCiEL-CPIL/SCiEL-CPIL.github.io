# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin (limit 60s)
    
    def get_subset_sums(elements):
        """Returns a set of all non-empty subset sums."""
        sums = {0}
        for x in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums.update(new_sums)
        sums.discard(0)
        return sums

    def is_valid(sums_set):
        """Check if no two sums divide each other."""
        # Optimization: sort sums. For each sum a, check if any multiple of a is in the set.
        # Since sums are positive, if b is a multiple of a and b != a, then b >= 2a.
        # So we only need to check multiples k*a for k >= 2.
        # To make this fast, we can iterate through sorted sums and for each a, check if any sum is a multiple of a.
        # Better: For each a in sums, check if any s in sums is divisible by a and s != a.
        # Even better: Sort sums. For each a, the smallest multiple of a greater than a is 2a.
        # So if 2a is in sums, invalid. If 3a is in sums, invalid, etc.
        # Actually, just checking if any s % a == 0 for s != a is sufficient.
        
        # Efficient check:
        # For each a in sums:
        #   for multiple in range(2*a, max_sum+1, a):
        #      if multiple in sums: return False
        # This is O(N * M / a).
        
        if not sums_set:
            return True
        
        max_sum = max(sums_set)
        # Sort to potentially break early? No, set lookup is O(1).
        # We can iterate through sums and check multiples.
        for a in sums_set:
            # Check multiples of a
            # Start from 2*a
            for k in range(2, max_sum // a + 1):
                if (a * k) in sums_set:
                    return False
        return True

    def try_add(x, current_sums, current_elements):
        """
        Attempts to add x to the set.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        current_sums is the set of subset sums of current_elements.
        """
        # New sums are current_sums U {x} U {s+x for s in current_sums}
        # Let S = current_sums.
        # New set S' = S U {x} U {s+x | s in S}
        
        # Check for collisions (duplicates)
        # 1. x must not be in S.
        if x in current_sums:
            return False, None
        
        # 2. x + s must not be in S for any s in S.
        #    x + s = t (t in S) => s = t - x.
        #    So if there exists t in S such that t - x is in S, then collision.
        #    Wait, x+s is a NEW sum. It collides with an OLD sum t if x+s = t.
        #    Or it collides with another NEW sum x+s' if x+s = x+s' => s=s' (unique).
        #    Or x collides with x+s? x = x+s => s=0 (0 not in S).
        #    So we just need to check if x+s is in S for any s in S.
        
        for s in current_sums:
            if (s + x) in current_sums:
                return False, None
        
        # Check if x collides with any s in S?
        # x is a new sum. It is in S? Already checked.
        # Does x divide any s in S or is x divisible by any s in S?
        # We will check divisibility on the full new set.
        
        # Construct new_sums
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        
        # Check validity of new_sums
        if not is_valid(new_sums):
            return False, None
            
        return True, new_sums

    def greedy_construct(ordering):
        """
        Greedy construction using a specific ordering of candidates.
        """
        elements = []
        sums = set()
        
        for x in ordering:
            if x in elements: # distinct integers
                continue
            ok, new_sums = try_add(x, sums, elements)
            if ok:
                elements.append(x)
                sums = new_sums
                
        return elements, sums

    def local_search(initial_elements, initial_sums):
        """
        Local search to improve the set size.
        """
        current_elements = list(initial_elements)
        current_sums = set(initial_sums)
        best_elements = list(current_elements)
        best_sums = set(current_sums)
        
        # Precompute candidates
        all_candidates = list(range(1, 257))
        
        # Helper to check if we can add x
        def can_add(x):
            if x in current_elements:
                return False
            return try_add(x, current_sums, current_elements)[0]
            
        # Helper to remove x
        def remove(x):
            # Recalculate sums from scratch is O(2^k). k is small.
            # But we can do better if we maintain a list.
            # For now, recalculation is simple and safe.
            new_elements = [e for e in current_elements if e != x]
            new_sums = get_subset_sums(new_elements)
            return new_elements, new_sums

        # Loop until deadline
        while time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add a random element
            # Sort candidates by "difficulty" or just random?
            # Random is fine.
            random.shuffle(all_candidates)
            added = False
            for x in all_candidates:
                if x in current_elements:
                    continue
                ok, new_sums = try_add(x, current_sums, current_elements)
                if ok:
                    current_elements.append(x)
                    current_sums = new_sums
                    if len(current_elements) > len(best_elements):
                        best_elements = list(current_elements)
                        best_sums = set(current_sums)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. If no add possible, try a swap
            # Remove one element, try to add one (or two?)
            # Strategy: Remove x, try to add y. If successful, size is same.
            # But maybe removing x allows adding y AND z?
            
            # Pick a random element to remove
            if not current_elements:
                break
                
            idx = random.randint(0, len(current_elements) - 1)
            x_remove = current_elements[idx]
            
            new_elements, new_sums = remove(x_remove)
            
            # Try to add candidates to the reduced set
            # We want to add at least 1 element to restore size, ideally 2 to increase size.
            # Let's try to add as many as possible greedily from the reduced set.
            # This is effectively restarting greedy on the reduced set + new candidates.
            
            # To be efficient, we can just try to add random candidates to new_sums.
            # If we add 1, size is same. If we add 2, size increases.
            
            # Let's try a few random swaps
            for _ in range(10): # Limit swaps per iteration to avoid stalling
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random candidate to add to the reduced set
                # We should prioritize candidates not in new_elements
                available = [x for x in all_candidates if x not in new_elements]
                if not available:
                    break
                x_add = random.choice(available)
                
                ok, new_sums_after_add = try_add(x_add, new_sums, new_elements)
                if ok:
                    # We have a valid set of size len(current_elements)
                    # Let's see if we can add MORE
                    final_elements = new_elements + [x_add]
                    final_sums = new_sums_after_add
                    
                    # Try to extend this new set
                    # This is a mini-greedy
                    random.shuffle(all_candidates)
                    for x_next in all_candidates:
                        if x_next in final_elements:
                            continue
                        ok2, new_sums2 = try_add(x_next, final_sums, final_elements)
                        if ok2:
                            final_elements.append(x_next)
                            final_sums = new_sums2
                            if len(final_elements) > len(best_elements):
                                best_elements = list(final_elements)
                                best_sums = set(final_sums)
                                improved = True
                                # Update current state to this improved state
                                current_elements = final_elements
                                current_sums = final_sums
                                break
                    if improved:
                        break
                    else:
                        # Even if not improved in size, we have a different valid set of same size.
                        # We can adopt it to escape local minima.
                        current_elements = final_elements
                        current_sums = final_sums
                        break
            
    # Main logic
    best_elements = []
    best_sums = set()
    
    # Multiple random restarts with different heuristics
    # Heuristic 1: Random
    # Heuristic 2: Descending (Large numbers are "safer" as they have fewer multiples in 1..256)
    # Heuristic 3: Ascending
    # Heuristic 4: Prime-like (numbers with few divisors)
    
    strategies = [
        lambda: list(range(1, 257)), # Ascending
        lambda: list(range(256, 0, -1)), # Descending
        lambda: (lambda x: (random.shuffle(x), x)[1])(list(range(1, 257))), # Random
        lambda: sorted(range(1, 257), key=lambda x: random.random()), # Random
    ]
    
    # Add a heuristic based on "isolation" - numbers that don't divide or are divided by many numbers in 1..256
    # Actually, large numbers are generally better.
    # Let's try a mix.
    
    for i in range(20): # 20 restarts
        if time.time() > deadline - 5:
            break
            
        # Choose strategy
        if i % 2 == 0:
            order = list(range(256, 0, -1))
        else:
            order = list(range(1, 257))
            random.shuffle(order)
            
        # Refine order: prioritize numbers that are "hard" to divide?
        # Actually, just running greedy on these orders is a good start.
        
        elements, sums = greedy_construct(order)
        
        if len(elements) > len(best_elements):
            best_elements = elements
            best_sums = sums
            
        # Local search on this result
        local_search(best_elements, best_sums)
        
        if len(best_elements) > len(best_elements): # Check if local search improved it
            pass # best_elements is already updated inside local_search if it found better
            
    # Final verification and cleanup
    # Ensure types are int
    result = [int(x) for x in best_elements]
    
    # One last check: is it valid?
    # If not, fallback to a known small valid set?
    # The problem says "Any broken rule makes the whole result invalid".
    # We must ensure validity.
    
    final_sums = get_subset_sums(result)
    if not is_valid(final_sums):
        # Fallback: simple greedy descending which is very likely valid
        fallback_elements = []
        fallback_sums = set()
        for x in range(256, 0, -1):
            ok, new_sums = try_add(x, fallback_sums, fallback_elements)
            if ok:
                fallback_elements.append(x)
                fallback_sums = new_sums
        result = [int(x) for x in fallback_elements]
        
    return result

# EVOLVE-BLOCK-END
