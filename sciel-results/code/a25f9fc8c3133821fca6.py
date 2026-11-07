import random
import time

def _is_valid(S):
    n = len(S)
    if n == 0:
        return True
    if len(set(S)) != n:
        return False
    for x in S:
        if not (1 <= x <= 256):
            return False
            
    sums = set()
    # Generate all non-empty subset sums
    # Using iterative method for speed
    sums.add(0)
    for x in S:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums or ns in new_sums:
                # Collision check can be done later, but early exit is good
                # Actually, if ns is already in sums, we have a collision
                # But we need to be careful: sums contains 0 and previous sums.
                # If ns is in sums, it means a subset of previous elements has sum ns.
                # So {subset} and {subset + x} have same sum. Invalid.
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    
    # Now sums contains all subset sums (including 0)
    # Remove 0 for divisibility check? 
    # Task says "non-empty subset sum".
    # So we check non-zero sums.
    non_empty_sums = [s for s in sums if s > 0]
    
    # Check divisibility
    # Sort sums to check multiples
    # s1 | s2 implies s2 = k * s1
    # We can just check for each s1, if 2*s1, 3*s1... is in sums
    max_sum = max(non_empty_sums)
    is_sum = [False] * (max_sum + 1)
    for s in non_empty_sums:
        is_sum[s] = True
        
    for s in non_empty_sums:
        for k in range(2, max_sum // s + 1):
            if is_sum[s * k]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    best_set = [31, 32, 34]
    
    # Try to find larger sets
    # Sizes 4 to 15
    for target_size in range(4, 16):
        # Try random restarts
        found = False
        for _ in range(5000):
            if time.time() > deadline:
                break
            S = random.sample(range(1, 257), target_size)
            if _is_valid(S):
                best_set = S
                found = True
                break
        if not found:
            # If we didn't find one, maybe it's hard. 
            # But we can still try next size? 
            # Or maybe we should stop if size is large and we've spent too much time?
            # Let's just continue to next size, but maybe reduce iterations?
            # Actually, if size 10 is not found in 5000 tries, size 11 is likely not found.
            # But random sampling might be bad.
            # Let's just proceed.
            pass
            
    return best_set
