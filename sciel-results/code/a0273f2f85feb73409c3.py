import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    # 90s limit, keep 5s safety margin
    deadline = start_time + 85.0
    
    # Fast subset sum computation and validity check
    def compute_sums(elements):
        """Compute all non-empty subset sums. Returns set or None if duplicates."""
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return None
                new_sums.add(ns)
            if x in sums:
                return None
            new_sums.add(x)
            sums = sums.union(new_sums)
        return sums
        
    def check_divisibility(sums):
        """Check no two distinct sums divide each other. Returns True if valid."""
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i+1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True

    def is_valid(elements):
        if not elements:
            return True
        if len(elements) != len(set(elements)):
            return False
        if any(x < 1 or x > 256 for x in elements):
            return False
        sums = compute_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)

    def can_add(x, current_elements, current_sums):
        """
        Check if x can be added to current_elements (which is valid).
        Returns (new_sums, True) if valid, (None, False) otherwise.
        """
        if x in current_elements:
            return None, False
        
        # 1. Check for sum collisions
        new_sums = set()
        if x in current_sums:
            return None, False
        new_sums.add(x)
        
        for s in current_sums:
            ns = s + x
            if ns in current_sums or ns in new_sums:
                return None, False
            new_sums.add(ns)
            
        # 2. Check divisibility
        # Only need to check pairs involving at least one new sum
        ns_list = sorted(new_sums)
        
        # Internal new sums
        for i in range(len(ns_list)):
            si = ns_list[i]
            for j in range(i+1, len(ns_list)):
                sj = ns_list[j]
                if sj % si == 0:
                    return None, False
        
        # New vs Old
        for sn in new_sums:
            for sc in current_sums:
                if sn > sc:
                    if sn % sc == 0:
                        return None, False
                else:
                    if sc % sn == 0:
                        return None, False
        
        return current_sums.union(new_sums), True

    def greedy_build(candidates, current_elements, current_sums, time_limit):
        """
        Greedily add elements from candidates to current_elements.
        Returns (elements, sums) of the best found.
        """
        for x in candidates:
            if time.time() > time_limit:
                break
            new_sums, valid = can_add(x, current_elements, current_sums)
            if valid:
                current_elements.append(x)
                current_sums = new_sums
        return current_elements, current_sums

    best_set = []
    best_size = 0
    all_candidates = list(range(1, 257))
    
    # Strategy: 
    # 1. Initial greedy with large numbers first (bias towards high values)
    # 2. Iterative improvement: 
    #    a. Try to add any remaining element.
    #    b. If stuck, perform a "kick": remove k random elements, then greedily refill.
    
    # Phase 1: Initial Construction
    # Bias towards larger numbers as they are less likely to divide smaller sums
    # and their multiples are less likely to be in the set [1, 256]
    init_candidates = sorted(all_candidates, reverse=True)
    random.shuffle(init_candidates) # Add some randomness to order
    
    curr_elems, curr_sums = greedy_build(init_candidates, [], set(), deadline - 5.0)
    
    if len(curr_elems) > best_size:
        best_size = len(curr_elems)
        best_set = list(curr_elems)
        
    # Phase 2: Local Search / Kicks
    # We try to improve the best set found.
    # If we can't add anything, we remove some elements to see if a different combination allows more.
    
    # Keep a set of elements for fast lookup
    best_set_set = set(best_set)
    
    # We will run this loop until time runs out
    # Each iteration: 
    # 1. Try to add any single element from [1,256] not in best_set.
    # 2. If none can be added, try a "kick": remove 1-3 random elements from best_set, 
    #    then try to greedily add elements back (including the removed ones and others).
    
    # Precompute sums for best_set
    best_sums = compute_sums(best_set)
    
    # List of available elements
    available = [x for x in range(1, 257) if x not in best_set_set]
    
    # To make "add any" faster, we don't check all 256 every time if we just did it.
    # We will re-check periodically or when the set changes.
    
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        
        # 1. Try to add a random available element
        # Instead of checking all, pick a few random ones to save time
        # Or check all if the set is small?
        # Let's try a random sample of available elements first.
        
        if available:
            # Try up to 10 random additions
            added = False
            for _ in range(10):
                if time.time() > deadline - 2.0: break
                if not available: break
                idx = random.randint(0, len(available) - 1)
                x = available[idx]
                
                new_sums, valid = can_add(x, best_set, best_sums)
                if valid:
                    # Accept
                    best_set.append(x)
                    best_set_set.add(x)
                    best_sums = new_sums
                    best_size += 1
                    available.remove(x)
                    added = True
                    # If we added, we might be able to add more, so continue inner loop
                    # But we broke the "random 10" loop logic? No, we continue the for loop.
                    # Actually, if we add one, the "available" list changes, and best_sums changes.
                    # So we should restart the "try to add" phase.
                    break
                    
            if added:
                continue # Go back to top of while loop to try adding more
        
        # 2. If no element could be added (or we didn't try enough), do a KICK
        # Remove k random elements to break the deadlock
        # k is usually small, like 1 or 2
        
        if len(best_set) > 3:
            # Choose number of elements to remove
            # Removing 1 is a standard move. Removing 2 allows more flexibility.
            # Let's try removing 1 first, then 2 if stuck for a while.
            num_remove = 1 if attempts % 2 == 0 else 2
            
            if len(best_set) > num_remove:
                # Pick random indices to remove
                remove_indices = random.sample(range(len(best_set)), num_remove)
                # Sort indices to remove safely
                remove_indices.sort(reverse=True)
                
                for idx in remove_indices:
                    x = best_set.pop(idx)
                    best_set_set.remove(x)
                    available.append(x)
                    
                # Recompute sums for the reduced set
                best_sums = compute_sums(best_set)
                best_size = len(best_set)
                
                # Now, greedily try to add elements back
                # We want to add back the removed ones, but also potentially others.
                # Shuffle available to mix them up
                random.shuffle(available)
                
                # Greedy refill
                # We try to add elements from available.
                # We prioritize the ones we just removed? Or just random order?
                # Random order is fine.
                
                for x in available:
                    if time.time() > deadline - 2.0:
                        break
                    new_sums, valid = can_add(x, best_set, best_sums)
                    if valid:
                        best_set.append(x)
                        best_set_set.add(x)
                        best_sums = new_sums
                        best_size += 1
                        # Remove from available
                        # Using list remove is O(N). For N=256, it's fine.
                        try:
                            available.remove(x)
                        except ValueError:
                            pass # Should not happen

    # Final validation
    if best_set:
        if not is_valid(best_set):
            # Fallback
            best_set = [100, 101]
            
    # Ensure types
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique
    
    return best_set

