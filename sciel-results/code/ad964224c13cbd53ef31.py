import time
import random

def is_valid(elements):
    """
    Returns True if the set of elements is valid.
    Valid means:
    1. All non-empty subset sums are distinct.
    2. No non-empty subset sum divides another non-empty subset sum.
    """
    n = len(elements)
    if n == 0:
        return True
    
    # Generate all subset sums
    sums = [0]
    for e in elements:
        # Add e to all existing sums
        sums += [s + e for s in sums]
        
    # Remove the empty set sum (0)
    sums = [s for s in sums if s != 0]
    
    # Check for duplicates
    if len(sums) != len(set(sums)):
        return False
        
    sums_set = set(sums)
    sums_list = sorted(sums_set)
    max_sum = sums_list[-1]
    
    # Check divisibility
    # For each a in sums_list, check if any b in sums_list (b > a) is divisible by a
    # Optimization: if a * 2 > max_sum, no multiple can exist in the set (except a itself, which is not distinct)
    for a in sums_list:
        if a * 2 > max_sum:
            continue
        # Check multiples of a
        multiple = a * 2
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += a
            
    return True

def construct():
    deadline = time.time() + 55.0 # Safety margin for 60s limit
    best_set = []
    best_size = 0
    
    # Initial seed: try to find a valid set of size 1, 2, 3...
    # Large numbers are generally better for divisibility constraints.
    # Let's start with a few random large numbers and validate.
    
    def random_valid_set(target_size):
        """Tries to find a valid set of exactly target_size."""
        for _ in range(1000):
            if time.time() > deadline:
                return None
            # Pick random distinct integers from 1..256
            # Bias towards larger numbers?
            # Let's just pick randomly first.
            candidates = random.sample(range(1, 257), target_size)
            if is_valid(candidates):
                return candidates
        return None

    # First, ensure we have a valid set of size 1
    # [1] is valid? Sums: {1}. No pairs. Valid.
    current_set = [1]
    best_set = [1]
    best_size = 1
    
    # Try to grow the set
    # We will try to add elements to current_set.
    # If we fail to add for a while, we might restart with a different base.
    
    # A more effective strategy:
    # 1. Find max possible size by trying to build sets of size k, k+1...
    # 2. Use local search to improve.
    
    # Let's try to find the largest possible set size by increasing k.
    # Start with k=1.
    found_size = 1
    current = [random.randint(100, 256)] # Start with a large number
    if not is_valid(current):
        current = [1]
    best_set = current[:]
    best_size = 1
    
    # Try to increase size
    for target_size in range(2, 15):
        if time.time() > deadline:
            break
            
        # Try to find a valid set of this size
        # Strategy: Take best_set (size target_size-1) and try to add/swap
        found = False
        
        # Attempt 1: Direct addition
        # Try random elements to add to best_set
        attempts = 0
        while attempts < 500 and time.time() < deadline:
            attempts += 1
            x = random.randint(1, 256)
            if x in best_set:
                continue
            new_set = best_set + [x]
            if is_valid(new_set):
                best_set = new_set
                best_size = target_size
                found = True
                break
                
        if found:
            continue # Try next size
            
        # Attempt 2: Local Search / Swap
        # If direct addition failed, try to swap elements in best_set to make room for a new one
        # Or just restart the search for this size with random sets
        # Given the small domain (256), random restarts for small k might work.
        
        # Let's try random restarts for this specific size
        for _ in range(200):
            if time.time() > deadline:
                break
            candidates = random.sample(range(1, 257), target_size)
            if is_valid(candidates):
                best_set = candidates
                best_size = target_size
                found = True
                break
                
        if not found:
            # We couldn't find a valid set of this size in the allotted attempts
            # It might be impossible or just hard to find.
            # We should probably stop increasing size if we've tried hard.
            # But maybe a different configuration works.
            # Let's do a more intensive local search from the best known set.
            
            # Local Search: Start with best_set, try to replace elements
            current = best_set[:]
            for _ in range(1000):
                if time.time() > deadline:
                    break
                # Pick a random index to replace
                idx = random.randint(0, len(current) - 1)
                old_val = current[idx]
                new_val = random.randint(1, 256)
                if new_val == old_val or new_val in current:
                    continue
                
                current[idx] = new_val
                if is_valid(current):
                    # Now try to add a new element to this valid set of size target_size-1
                    # Wait, current is size target_size-1. We want size target_size.
                    # So we need to add one.
                    # Actually, if we swapped, size is still target_size-1.
                    # We need to find a valid set of size target_size.
                    # Let's try to add to the swapped current.
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        add_val = random.randint(1, 256)
                        if add_val in current:
                            continue
                        candidate = current + [add_val]
                        if is_valid(candidate):
                            best_set = candidate
                            best_size = target_size
                            found = True
                            break
                    if found:
                        break
                else:
                    current[idx] = old_val # Revert if invalid? Or keep invalid state?
                    # Keeping invalid state allows "walking" through invalid space, but is_valid is expensive.
                    # Better to revert.

            if not found:
                # If we still haven't found it, assume we've hit the limit for this seed strategy
                # or it's very hard. Break to save time for validation/finalization?
                # Or continue to next size? No, if we can't make size K, we can't make K+1.
                # But maybe a different random restart works.
                # Let's break out of the size loop if we've tried hard.
                break

    # Final validation
    if not is_valid(best_set):
        # Fallback to a known small valid set
        # [20, 21, 23] is valid (size 3)
        best_set = [20, 21, 23]
        
    return list(best_set)
