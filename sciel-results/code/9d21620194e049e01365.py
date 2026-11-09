import time
import random
from itertools import combinations

# LCM of 1..22
_LCM = 232792560

def _is_valid_fast(subset):
    """
    Fast check if all subset sums of reciprocals are distinct.
    Uses incremental collision detection with early termination.
    """
    if not subset:
        return True
    
    # Convert to integer weights, sort in ascending order for better early termination
    # (smaller weights first means we detect collisions earlier when sums overlap)
    weights = sorted([_LCM // i for i in subset])
    k = len(weights)
    
    # Generate all subset sums incrementally, check for collisions
    sums = {0}
    for w in weights:
        # Add w to all existing sums
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
        
    return True

def _is_valid_with_existing_sums(subset, existing_sums):
    """
    Check if adding subset to existing_sums creates any collisions.
    Returns (valid, new_sums) or (False, None).
    """
    if not subset:
        return True, existing_sums
    
    # Convert to integer weights, sort in ascending order
    weights = sorted([_LCM // i for i in subset])
    
    # Start with existing sums
    sums = set(existing_sums)
    
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False, None
            new_sums.add(ns)
        sums |= new_sums
        
    return True, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s limit, leave buffer)
    
    # Precompute weights for all numbers 1..22
    all_weights = {i: _LCM // i for i in range(1, 23)}
    
    # Start with a known good seed
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)
    
    # Strategy 1: Systematic greedy construction with multiple random restarts
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Different strategies for ordering
        if iteration % 4 == 0:
            # Random order
            candidates = list(range(1, 23))
            random.shuffle(candidates)
        elif iteration % 4 == 1:
            # Prefer larger numbers (smaller weights) first
            candidates = list(range(22, 0, -1))
            random.shuffle(candidates)
        elif iteration % 4 == 2:
            # Prefer smaller numbers (larger weights) first
            candidates = list(range(1, 23))
            random.shuffle(candidates)
        else:
            # Sort by weight (descending) with some randomness
            candidates = list(range(1, 23))
            candidates.sort(key=lambda x: all_weights[x], reverse=True)
            # Add randomness by shuffling adjacent elements
            for i in range(1, len(candidates)):
                if random.random() < 0.3:
                    candidates[i-1], candidates[i] = candidates[i], candidates[i-1]
        
        current = []
        current_set = set()
        
        for c in candidates:
            if c in current_set:
                continue
            # Try to add
            if _is_valid_fast(current + [c]):
                current.append(c)
                current_set.add(c)
        
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)
            # Try to extend even more
            remaining = [x for x in range(1, 23) if x not in current_set]
            for c in remaining:
                if time.time() > deadline:
                    break
                if _is_valid_fast(current + [c]):
                    current.append(c)
                    current_set.add(c)
            if len(current) > best_size:
                best_size = len(current)
                best_subset = list(current)

    # Strategy 2: Local search - try to improve best_subset by adding elements
    if time.time() < deadline - 5:
        best_set = set(best_subset)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for c in range(1, 23):
                if c not in best_set:
                    if _is_valid_fast(list(best_set) + [c]):
                        best_set.add(c)
                        improved = True
                        if len(best_set) > best_size:
                            best_size = len(best_set)
                            best_subset = list(best_set)
                        break

    # Strategy 3: Try to find larger sets by removing one element and adding two
    if time.time() < deadline - 3:
        best_set = set(best_subset)
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Try removing one element and adding two different ones
            found = False
            for remove_c in list(best_set):
                if time.time() > deadline:
                    break
                temp_set = set(best_set)
                temp_set.remove(remove_c)
                
                # Try adding pairs of elements
                for add1 in range(1, 23):
                    if add1 in temp_set:
                        continue
                    for add2 in range(add1+1, 23):
                        if add2 in temp_set:
                            continue
                        candidate = list(temp_set) + [add1, add2]
                        if len(candidate) > len(best_set):
                            if _is_valid_fast(candidate):
                                best_set = set(candidate)
                                found = True
                                if len(best_set) > best_size:
                                    best_size = len(best_set)
                                    best_subset = list(best_set)
                                break
                    if found:
                        break
                if found:
                    break

    return best_subset
