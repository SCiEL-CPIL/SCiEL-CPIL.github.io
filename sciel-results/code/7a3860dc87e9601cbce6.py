"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 256, 'maximum_size': 60}

def _is_valid(arr):
    """
    Check if the set satisfies the constraint: different cardinalities have different sums.
    Uses bitset DP for efficiency.
    """
    n = len(arr)
    if n == 0:
        return True
    
    max_sum = sum(arr)
    # sums_by_card[k] is a bitset (integer) where bit s is 1 if sum s is achievable with k elements
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1  # Bit 0 is set for sum 0 with 0 elements
    
    for x in arr:
        # Iterate backwards to avoid using the same element twice
        for k in range(n - 1, -1, -1):
            # Shift the bitset of sums for k elements by x and OR into k+1
            # We only care about sums up to max_sum
            sums_by_card[k + 1] |= (sums_by_card[k] << x)
            # Optional: mask to keep integers small, though Python handles big ints well
            # sums_by_card[k + 1] &= (1 << (max_sum + 1)) - 1
            
    # Check for disjointness between all pairs of cardinalities
    for k1 in range(n + 1):
        if sums_by_card[k1] == 0:
            continue
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k2] == 0:
                continue
            if sums_by_card[k1] & sums_by_card[k2]:
                return False
    return True

def _count_collisions(arr):
    """Count the number of cross-cardinality sum collisions."""
    n = len(arr)
    if n == 0:
        return 0
        
    max_sum = sum(arr)
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    
    for x in arr:
        for k in range(n - 1, -1, -1):
            sums_by_card[k + 1] |= (sums_by_card[k] << x)
            
    collisions = 0
    for k1 in range(n + 1):
        if sums_by_card[k1] == 0:
            continue
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k2] == 0:
                continue
            # Count bits in intersection
            intersection = sums_by_card[k1] & sums_by_card[k2]
            collisions += bin(intersection).count('1')
    return collisions

def _greedy_construct(deadline):
    """
    Greedy construction: try to add the smallest valid integer at each step.
    """
    result = []
    for candidate in range(1, 257):
        if time.time() > deadline:
            break
        test_set = result + [candidate]
        if _is_valid(test_set):
            result.append(candidate)
            if len(result) >= P['maximum_size']:
                break
    return result

def _random_search_with_removal(deadline):
    """
    Start with a large random set and iteratively remove elements involved in the most collisions.
    """
    # Start with a large random subset
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    current_set = candidates[:60]
    best_set = list(current_set)
    best_size = len(current_set)
    
    # Try to improve by removing elements with most collisions
    for iteration in range(200): # Reduced iterations due to faster validation
        if time.time() > deadline:
            break
        
        if not _is_valid(current_set):
            collisions = _count_collisions(current_set)
            if collisions == 0:
                break
            
            min_collisions = collisions
            best_remove = -1
            n = len(current_set)
            
            # Heuristic: try removing each element and see if collisions decrease
            for idx in range(n):
                test_set = current_set[:idx] + current_set[idx+1:]
                test_collisions = _count_collisions(test_set)
                if test_collisions < min_collisions:
                    min_collisions = test_collisions
                    best_remove = idx
            
            if best_remove >= 0:
                current_set = current_set[:best_remove] + current_set[best_remove+1:]
            else:
                # No single removal helps; break
                break
        
        # Try to add elements back
        added = False
        for candidate in range(1, 257):
            if time.time() > deadline:
                break
            if candidate not in current_set:
                test_set = current_set + [candidate]
                if _is_valid(test_set):
                    current_set = test_set
                    added = True
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        if best_size >= P['maximum_size']:
            break
        if not added and _is_valid(current_set):
             # If we are valid but couldn't add more, we are stuck
             break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10 second safety margin
    
    # First, try greedy construction which is fast and reliable
    greedy_result = _greedy_construct(deadline)
    best_result = list(greedy_result)
    best_size = len(best_result)
    
    # If greedy didn't reach maximum, try random search with removal
    if best_size < P['maximum_size'] and time.time() < deadline:
        random.seed(42)
        random_result = _random_search_with_removal(deadline)
        if len(random_result) > best_size:
            best_result = random_result
            best_size = len(best_result)
    
    # Final validation to ensure we don't return an invalid set
    if not _is_valid(best_result):
        # Fallback: return a known valid small set or empty if nothing works
        # A single element [1] is always valid
        return [1]
        
    # Ensure we return a valid list of Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
