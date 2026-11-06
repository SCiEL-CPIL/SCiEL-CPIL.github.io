import time
import random
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    # Safe base set: (2n/3, n]
    base_start = 171
    S = set(range(base_start, n + 1))
    
    # Precompute rem_groups for all a: for each a, dict rem -> sorted list of x in (a, n] with x%a == rem
    rem_groups_cache = {}
    for a in range(1, n + 1):
        rem_groups = {}
        for x in range(a + 1, n + 1):
            r = x % a
            if r not in rem_groups:
                rem_groups[r] = []
            rem_groups[r].append(x)
        rem_groups_cache[a] = rem_groups
    
    def can_add(a, S):
        """Check if adding a to S violates the property."""
        # For each b in S with b > a, check if there exists c in S with c > b and c % a == (-b) % a
        # Build per-remainder lists for S elements > a
        s_rem_lists = {}
        for x in S:
            if x > a:
                r = x % a
                if r not in s_rem_lists:
                    s_rem_lists[r] = []
                s_rem_lists[r].append(x)
        
        # Sort each list
        for r in s_rem_lists:
            s_rem_lists[r].sort()
        
        # For each b in S with b > a, check conflict
        for b in S:
            if b <= a:
                continue
            target_rem = (-b) % a
            if target_rem in s_rem_lists:
                group = s_rem_lists[target_rem]
                idx = bisect_left(group, b + 1)
                if idx < len(group):
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
    
    # Swap-based local search: try removing one element to allow adding two others
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            S_list = sorted(S)
            not_in_S = [a for a in range(1, n + 1) if a not in S]
            random.shuffle(not_in_S)
            
            # Try each possible removal
            for rem_elem in S_list:
                if time.time() > deadline - 2.0:
                    break
                # Try removing rem_elem
                S_test = S - {rem_elem}
                added_count = 0
                added_elems = []
                
                for add_elem in not_in_S:
                    if time.time() > deadline - 2.0:
                        break
                    if can_add(add_elem, S_test):
                        S_test.add(add_elem)
                        added_elems.append(add_elem)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count > 1:
                    S = S_test
                    improved = True
                    break
            
            # Also try 2-for-3 swaps
            if not improved and time.time() < deadline - 3.0:
                S_list2 = sorted(S)
                not_in_S2 = [a for a in range(1, n + 1) if a not in S]
                random.shuffle(not_in_S2)
                
                # Try removing two elements
                for i in range(min(len(S_list2), 50)):
                    if time.time() > deadline - 3.0:
                        break
                    for j in range(i + 1, min(len(S_list2), 50)):
                        if time.time() > deadline - 3.0:
                            break
                        S_test = S - {S_list2[i], S_list2[j]}
                        added_count = 0
                        added_elems = []
                        
                        for add_elem in not_in_S2:
                            if time.time() > deadline - 3.0:
                                break
                            if can_add(add_elem, S_test):
                                S_test.add(add_elem)
                                added_elems.append(add_elem)
                                added_count += 1
                                if added_count >= 3:
                                    break
                        
                        if added_count > 2:
                            S = S_test
                            improved = True
                            break
                    if improved:
                        break
    
    # Final pass
    if time.time() < deadline - 1.0:
        remaining3 = [a for a in range(1, n + 1) if a not in S]
        random.shuffle(remaining3)
        for a in remaining3:
            if time.time() > deadline - 0.5:
                break
            if can_add(a, S):
                S.add(a)
    
    # One more round of 1-for-2 swaps with fresh randomization
    if time.time() < deadline - 1.5:
        S_list_final = sorted(S)
        not_in_S_final = [a for a in range(1, n + 1) if a not in S]
        random.shuffle(not_in_S_final)
        
        for rem_elem in S_list_final:
            if time.time() > deadline - 1.0:
                break
            S_test = S - {rem_elem}
            added_count = 0
            added_elems = []
            
            for add_elem in not_in_S_final:
                if time.time() > deadline - 1.0:
                    break
                if can_add(add_elem, S_test):
                    S_test.add(add_elem)
                    added_elems.append(add_elem)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count > 1:
                S = S_test
    
    return list(S)
