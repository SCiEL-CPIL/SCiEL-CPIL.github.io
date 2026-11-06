import time
import random
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    # Safe base set: (2n/3, n]
    base_start = 171
    S = set(range(base_start, n + 1))
    
    def can_add(a, S):
        # Get elements in S greater than a, sorted
        elems = sorted(x for x in S if x > a)
        if len(elems) < 2:
            return True
        
        # Group elements by their remainder mod a
        rem_groups = {}
        for e in elems:
            r = e % a
            rem_groups.setdefault(r, []).append(e)
        
        # For each b in elems, check if there exists c > b with c % a == (-b) % a
        for b in elems:
            target_rem = (-b) % a
            if target_rem in rem_groups:
                # Need c > b with c % a == target_rem
                # Binary search for first element > b in rem_groups[target_rem]
                lst = rem_groups[target_rem]
                idx = bisect_right(lst, b)
                if idx < len(lst):
                    return False
        return True

    def try_add(a, S):
        if a in S:
            return False
        if can_add(a, S):
            S.add(a)
            return True
        return False

    # Greedy Phase: Add from top to bottom
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline:
            break
        try_add(a, S)
    
    # Local Search: Try to add remaining elements in random order
    remaining = [a for a in range(1, base_start) if a not in S]
    random.shuffle(remaining)
    
    for a in remaining:
        if time.time() > deadline:
            break
        try_add(a, S)
    
    # Multiple passes to catch order-dependent additions
    for _ in range(5):
        if time.time() > deadline:
            break
        remaining2 = [a for a in range(1, n + 1) if a not in S]
        if not remaining2:
            break
        random.shuffle(remaining2)
        for a in remaining2:
            if time.time() > deadline:
                break
            try_add(a, S)
    
    # Swap-based local search: try removing one element and adding two others
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline:
                break
            S_list = list(S)
            if len(S_list) < 3:
                break
            # Pick a random element to potentially remove
            i = random.randint(0, len(S_list) - 1)
            elem_to_remove = S_list[i]
            
            # Try removing it and see if we can add more
            S_temp = S - {elem_to_remove}
            added_count = 0
            
            remaining3 = [a for a in range(1, n + 1) if a not in S_temp and a != elem_to_remove]
            random.shuffle(remaining3)
            
            for a in remaining3:
                if time.time() > deadline:
                    break
                if a in S_temp:
                    continue
                if can_add(a, S_temp):
                    S_temp.add(a)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            # If we added at least 2 elements, accept the swap
            if added_count >= 2 and len(S_temp) > len(S):
                S = S_temp
    
    # Final polish: another pass with different ordering
    if time.time() < deadline - 5:
        remaining4 = [a for a in range(1, n + 1) if a not in S]
        random.shuffle(remaining4)
        for a in remaining4:
            if time.time() > deadline:
                break
            try_add(a, S)
    
    return list(S)
