import time
import random

START_TIME = time.time()
TIME_LIMIT = 85.0

def _check_divisibility_fast(sums_set):
    """
    Check if any two distinct sums in the set have a divisibility relation.
    Assumes sums_set contains unique positive integers.
    Returns True if valid (no divisibility), False otherwise.
    """
    if len(sums_set) <= 1:
        return True
    
    sums_list = sorted(sums_set)
    max_sum = sums_list[-1]
    
    # To speed up, we can just iterate and check multiples
    # But checking if a multiple exists in a set is O(1)
    for s in sums_list:
        if s == 0:
            continue
        # Check multiples 2*s, 3*s, ...
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _can_add_fast(current_sums, new_elem):
    """
    Check if adding new_elem to the set corresponding to current_sums is valid.
    current_sums: set of all non-empty subset sums of the current set.
    Returns (is_valid, new_sums)
    """
    # Generate new sums
    new_sums = set()
    for s in current_sums:
        ns = s + new_elem
        new_sums.add(ns)
    new_sums.add(new_elem)
    
    # Check for collisions (distinct subset sums requirement)
    # If any new sum is already in current_sums, we have a collision.
    # Also, if new_sums has fewer elements than expected, there were internal collisions.
    # Expected new sums count: len(current_sums) + 1
    if len(new_sums) < len(current_sums) + 1:
        return False, None
    
    # Check intersection
    if current_sums.intersection(new_sums):
        return False, None
        
    # Combine
    all_sums = current_sums.union(new_sums)
    
    # Check divisibility
    if not _check_divisibility_fast(all_sums):
        return False, None
        
    return True, all_sums

def _greedy_construct(pool, rng):
    """
    Greedy construction with fast checks.
    """
    result = []
    current_sums = set()
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for elem in remaining:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        if len(result) >= 30:
            break
            
        is_valid, new_sums = _can_add_fast(current_sums, elem)
        if is_valid:
            result.append(elem)
            current_sums = new_sums
            
    return result

def construct():
    """
    Main entry point.
    """
    # We want large numbers.
    # Let's try a few strategies.
    # 1. Descending order (deterministic)
    # 2. Randomized descending (shuffle upper half)
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Pure Descending
    pool_desc = list(range(256, 0, -1))
    res1 = _greedy_construct(pool_desc, random.Random(1))
    if len(res1) > best_size:
        best_result = res1
        best_size = len(res1)
        
    # Strategy 2: Randomized attempts biased to large numbers
    # We'll run several random greedy attempts.
    # To bias towards large numbers, we can shuffle a range like [100, 256] and then [1, 99]
    # Or just shuffle the whole pool but the greedy will pick what it can.
    # Actually, greedy picks in order. So if we put large numbers first, it picks them first.
    
    attempts = 100
    for i in range(attempts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        rng = random.Random(i * 12345 + 6789)
        
        # Create a biased pool:
        # High numbers (128-256) are more likely to be compatible.
        # Let's shuffle 128-256 and put them first, then shuffle 1-127 and put them second.
        # This encourages picking large numbers.
        high = list(range(256, 127, -1))
        low = list(range(127, 0, -1))
        rng.shuffle(high)
        rng.shuffle(low)
        pool = high + low
        
        res = _greedy_construct(pool, rng)
        
        if len(res) > best_size:
            best_result = res
            best_size = len(res)
            
        # If we found a very large set, we might stop early or continue to see if we can do better
        if best_size >= 20:
             # We can stop if we are confident, but let's keep searching for max
             pass
             
    # If we didn't find anything (shouldn't happen, even 1 number is valid), fallback
    if not best_result:
        best_result = [1]
        
    return [int(x) for x in best_result]
