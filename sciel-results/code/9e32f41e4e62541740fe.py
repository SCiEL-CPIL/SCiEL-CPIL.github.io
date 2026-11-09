"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

# Precompute LCM of 1..22
_LCM = 232792560

def _can_add(sums_set, v):
    """
    Check if adding value v to the set of sums `sums_set` keeps all sums distinct.
    This is true if (sums_set + v) does not intersect with sums_set.
    """
    # If any s in sums_set is such that s + v is also in sums_set, then there is a collision.
    # We can check this by iterating over the smaller set.
    # Since sums_set grows exponentially, this is O(2^k).
    # Optimization: Just check intersection.
    # (sums_set + v) is a set of size |sums_set|.
    # We need (sums_set + v).intersection(sums_set) to be empty.
    # This is equivalent to: no s in sums_set such that s+v in sums_set.
    # We can iterate over sums_set.
    # To speed this up, we can break early.
    for s in sums_set:
        if s + v in sums_set:
            return False
    return True

def _get_sums(subset):
    """Compute the set of all subset sums for the given subset of indices."""
    if not subset:
        return {0}
    sums = {0}
    for i in subset:
        v = _LCM // i
        new_sums = {s + v for s in sums}
        sums = sums | new_sums
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    candidates = list(range(1, 23))
    values = {i: _LCM // i for i in candidates}
    
    best_set = []
    best_size = 0
    
    # Strategy: Randomized Greedy with Local Search (Add and Swap)
    
    # Run multiple restarts
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize with a random subset or empty
        # Let's try to build from scratch with random order
        order = candidates[:]
        rng.shuffle(order)
        
        current_set = []
        current_sums = {0}
        
        # Greedy Phase
        for i in order:
            v = values[i]
            if _can_add(current_sums, v):
                # Add i
                current_set.append(i)
                # Update sums
                # new_sums = {s + v for s in current_sums}
                # current_sums = current_sums | new_sums
                # To save time, we can update in place if we are careful, 
                # but creating a new set is safer and often fast enough for k<20.
                new_sums = {s + v for s in current_sums}
                current_sums = current_sums | new_sums
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Local Search Phase on current_set
        # Try to add missing elements
        # Try to swap existing with missing if it helps
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = [i for i in candidates if i not in current_set]
            
            # 1. Try to add any missing element
            for i in missing:
                v = values[i]
                if _can_add(current_sums, v):
                    current_set.append(i)
                    new_sums = {s + v for s in current_sums}
                    current_sums = current_sums | new_sums
                    improved = True
                    break
            
            if improved:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                continue
            
            # 2. If no element can be added, try swaps
            # Swap an existing element 'out' with a missing element 'in'
            # This is more complex. We need to recompute sums.
            # We can just try random swaps.
            if missing:
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    out_idx = rng.randrange(len(current_set))
                    out_elem = current_set[out_idx]
                    in_elem = rng.choice(missing)
                    
                    # Tentative swap
                    # Remove out_elem, add in_elem
                    # This requires rebuilding sums or a clever update.
                    # Rebuilding is O(2^k). k is at most 22. 2^22 is ~4M.
                    # Rebuilding 10 times per iteration might be slow if we do many iterations.
                    # But we only do this if we are stuck.
                    
                    new_set = current_set[:]
                    new_set[out_idx] = in_elem
                    
                    # Check if new_set is valid
                    # We can check if it's dissociated.
                    # Instead of full rebuild, we can check if the new set of sums has size 2^k.
                    # But we don't have the sums for the new set easily.
                    # Let's just verify the new set fully.
                    if _verify_recip_dissociated(new_set):
                        # If valid, accept the swap?
                        # The swap itself doesn't increase size.
                        # It might allow future additions.
                        # So we accept it to escape local minima.
                        current_set = new_set
                        # Recompute sums
                        current_sums = _get_sums(current_set)
                        improved = True
                        break
                        
    # Final local search on best_set to ensure it's maximal
    # Try to add any element
    final_improved = True
    while final_improved and time.time() < deadline:
        final_improved = False
        current_sums = _get_sums(best_set)
        for i in candidates:
            if i not in best_set:
                v = values[i]
                if _can_add(current_sums, v):
                    best_set.append(i)
                    best_size = len(best_set)
                    final_improved = True
                    break

    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set

def _verify_recip_dissociated(subset):
    """
    Return True if all 2^len(subset) reciprocal subset sums are distinct.
    Uses integer arithmetic with LCM as denominator.
    """
    if not subset:
        return True
    
    k = len(subset)
    values = [_LCM // i for i in subset]
    
    sums = {0}
    for v in values:
        new_sums = {s + v for s in sums}
        sums = sums | new_sums
        # Early exit if size is not doubling
        if len(sums) != (1 << (len([x for x in values if x <= v]) + 1 if False else len(sums) and 1)): # This logic is messy, just check at end or simple count
             pass
    return len(sums) == (1 << k)
