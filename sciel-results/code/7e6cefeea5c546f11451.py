import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    # Safe base set: (2n/3, n]
    base_start = 171
    S = set(range(base_start, n + 1))
    
    # Precompute for each a: the set of "bad sums" that would violate the condition
    # For a given a, sum s is bad if s % a == 0
    # We need to check if any pair (b,c) in S with a < b <= c has b+c in bad_sums[a]
    
    # Optimize can_add by checking pairs more efficiently
    def can_add(a, S):
        # Get elements in S greater than a, sorted
        elems = sorted(x for x in S if x > a)
        if len(elems) < 2:
            return True
        
        # For each pair, check if sum is divisible by a
        # Optimization: for a fixed b, we need c such that (b+c) % a == 0
        # i.e., c % a == (-b) % a
        # Group elements by their remainder mod a
        rem_groups = {}
        for e in elems:
            r = e % a
            rem_groups.setdefault(r, []).append(e)
        
        # For each b in elems, check if there exists c > b in same remainder class
        # where c % a == (-b) % a
        for b in elems:
            target_rem = (-b) % a
            if target_rem in rem_groups:
                # Need c > b with c % a == target_rem
                # Since elems is sorted, we can binary search or just check
                for c in rem_groups[target_rem]:
                    if c > b:
                        return False
        return True

    # Greedy Phase: Add from top to bottom
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    # Local Search: Try to add remaining elements in random order
    remaining = [a for a in range(1, base_start) if a not in S]
    random.shuffle(remaining)
    
    for a in remaining:
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    # Multiple passes to catch order-dependent additions
    for _ in range(3):
        if time.time() > deadline:
            break
        remaining2 = [a for a in range(1, n + 1) if a not in S]
        if not remaining2:
            break
        random.shuffle(remaining2)
        for a in remaining2:
            if time.time() > deadline:
                break
            if can_add(a, S):
                S.add(a)
    
    return list(S)
