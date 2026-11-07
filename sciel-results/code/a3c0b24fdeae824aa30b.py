import itertools
import random
import time

def _is_valid(els):
    n = len(els)
    # Generate all subset sums
    sums = {0}
    for e in els:
        # Update sums
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns not in sums:
                new_sums.add(ns)
            else:
                # Collision: ns is already in sums.
                # This means two different subsets sum to ns.
                # One is the existing one, one is current_subset + e.
                # So invalid.
                return False
        sums.update(new_sums)
    
    # Check if all sums are distinct (implicitly done by returning False on collision)
    # But wait, the above logic `if ns not in sums` only checks against previous sums.
    # If two different *new* sums collide with each other?
    # e.g. s1 + e == s2 + e => s1 == s2. Since s1, s2 are distinct in `sums`, this can't happen.
    # So the logic `if ns not in sums: add else: fail` is correct for uniqueness.
    
    # Now check divisibility
    # Remove 0
    if 0 in sums:
        sums.discard(0)
        
    sorted_sums = sorted(sums)
    max_sum = sorted_sums[-1] if sorted_sums else 0
    
    for s in sorted_sums:
        # Check if any multiple of s exists in sums
        # 2s, 3s, ...
        if s == 0: continue
        for k in range(2, max_sum // s + 1):
            if s * k in sums:
                return False
    return True

def _find_valid_set(max_val=256, target_size=12, timeout=55):
    start = time.time()
    random.seed(42)
    
    # Try decreasing sizes from a high target
    # We know size 8 is easy. Let's aim for 12.
    for n in range(target_size, 6, -1):
        if time.time() - start > timeout - 5:
            break
        max_attempts = 50000 if n <= 10 else 20000
        for attempt in range(max_attempts):
            if time.time() - start > timeout - 5:
                break
            # Bias towards larger numbers to reduce small factor multiples
            # Also ensure numbers are distinct
            cand = random.sample(range(100, max_val + 1), n)
            if _is_valid(cand):
                return cand
    # Fallback
    for n in range(6, 1, -1):
        if time.time() - start > timeout - 2:
            break
        for attempt in range(10000):
            if time.time() - start > timeout - 2:
                break
            cand = random.sample(range(1, max_val + 1), n)
            if _is_valid(cand):
                return cand
    return [31, 32, 34]

def construct():
    result = _find_valid_set()
    return [int(x) for x in result]
