import time
import random
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    base_start = 171
    S = set(range(base_start, n + 1))
    
    def can_add(a, S):
        elems = [x for x in S if x > a]
        if len(elems) < 2:
            return True
        
        rem_groups = {}
        for e in elems:
            r = e % a
            if r not in rem_groups:
                rem_groups[r] = []
            rem_groups[r].append(e)
        
        for r in rem_groups:
            if len(rem_groups[r]) > 1:
                rem_groups[r].sort()
        
        for b in elems:
            target_rem = (-b) % a
            if target_rem in rem_groups:
                group = rem_groups[target_rem]
                idx = bisect_right(group, b)
                if idx < len(group):
                    return False
        return True

    # 1. Greedy
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
            
    # 2. Random Add
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        remaining = [a for a in range(1, n + 1) if a not in S]
        random.shuffle(remaining)
        for a in remaining:
            if time.time() > deadline - 20:
                break
            if can_add(a, S):
                S.add(a)
                
    # 3. Swap Search (Lightweight)
    # We try to remove 1 and add 2.
    # To keep it fast, we only try a limited number of candidates for addition.
    
    while time.time() < deadline - 5.0:
        if len(S) < 3:
            break
            
        elem_to_remove = random.choice(list(S))
        S_temp = S - {elem_to_remove}
        
        # Get available elements
        available = [a for a in range(1, n + 1) if a not in S_temp]
        # Shuffle and take a subset to limit cost
        random.shuffle(available)
        # Limit checks to 100 candidates
        candidates = available[:100]
        
        added_count = 0
        for a in candidates:
            if time.time() > deadline - 5.0:
                break
            if can_add(a, S_temp):
                S_temp.add(a)
                added_count += 1
                if added_count >= 2:
                    break
        
        if len(S_temp) > len(S):
            S = S_temp
            
    # 4. Final Polish
    remaining_final = [a for a in range(1, n + 1) if a not in S]
    random.shuffle(remaining_final)
    for a in remaining_final:
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
            
    return list(S)
