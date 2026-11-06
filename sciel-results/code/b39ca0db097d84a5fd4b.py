import time
import random
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 40.0
    
    n = 256
    # Safe base set: (2n/3, n]
    # 2*256/3 = 170.66 -> 171
    base_start = 171
    S = set(range(base_start, n + 1))
    
    def can_add(a, S):
        # Count remainders of elements in S that are > a
        # If there exist b, c in S with b <= c and a | (b+c), it's invalid.
        # This happens if:
        # 1. There is an element x in S with x > a such that 2x is divisible by a. (Then b=c=x)
        # 2. There exist distinct x, y in S with x, y > a such that (x+y) is divisible by a.
        
        rem_counts = {}
        for x in S:
            if x > a:
                r = x % a
                rem_counts[r] = rem_counts.get(r, 0) + 1
        
        # Check condition 1: self-pair
        for r, count in rem_counts.items():
            if (2 * r) % a == 0:
                return False
        
        # Check condition 2: distinct pairs
        # We only need to check if there's a pair of remainders r1, r2 such that (r1+r2)%a==0
        # and both exist in rem_counts.
        # To avoid double checking, we can iterate and check complement.
        # Since a is small (<=256), we can just check all remainders present.
        present_rems = list(rem_counts.keys())
        for r1 in present_rems:
            r2 = (-r1) % a
            if r1 == r2:
                # Already handled by condition 1? 
                # If r1 == r2, then 2*r1 = 0 mod a. Condition 1 returns False if count >= 1.
                # So if we are here, condition 1 didn't trigger, meaning (2*r1)%a != 0 for all r1 in present_rems?
                # No, condition 1 iterates all. If (2*r1)%a==0, it returns False.
                # So if we reach here, no r has 2r=0 mod a. Thus r1 != r2 is guaranteed for valid pairs?
                # Wait. If (2*r1)%a != 0, then r1 != (-r1)%a. So r1 != r2.
                continue
            if r2 in rem_counts:
                return False
                
        return True

    def try_add(a, S):
        if a in S:
            return False
        if can_add(a, S):
            S.add(a)
            return True
        return False

    # Phase 1: Greedy Addition
    # Add elements from top to bottom.
    # Larger elements are 'easier' to add (fewer multiples in range).
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline - 15.0:
            break
        try_add(a, S)
    
    # Phase 2: Randomized Local Search (Additions)
    # Try to add remaining elements in random order to catch order-dependent opportunities.
    for _ in range(10):
        if time.time() > deadline - 15.0:
            break
        remaining = [a for a in range(1, base_start) if a not in S]
        if not remaining:
            break
        random.shuffle(remaining)
        for a in remaining:
            if time.time() > deadline - 15.0:
                break
            try_add(a, S)

    # Phase 3: Lightweight Local Search
    # Try to remove one element and add one or two others.
    # Limit iterations to ensure we finish in time.
    max_swaps = 50
    swaps_done = 0
    while time.time() < deadline - 2.0 and swaps_done < max_swaps:
        S_list = list(S)
        if len(S_list) < 3:
            break
        
        # Pick a random element to potentially remove
        i = random.randint(0, len(S_list) - 1)
        elem_to_remove = S_list[i]
        
        S_temp = S - {elem_to_remove}
        added_count = 0
        
        # Candidates to add: all elements not in S_temp
        # Sort candidates to try smaller ones first (often easier to fit?) or random.
        # Random is fine.
        remaining = [a for a in range(1, n + 1) if a not in S_temp]
        random.shuffle(remaining)
        
        # Try to add elements until we have added 2 or time runs out
        for a in remaining:
            if time.time() > deadline - 2.0:
                break
            if a in S_temp:
                continue
            if can_add(a, S_temp):
                S_temp.add(a)
                added_count += 1
                if added_count >= 2:
                    break
            
        # Accept the swap if the set size increased
        if len(S_temp) > len(S):
            S = S_temp
            swaps_done += 1
        else:
            swaps_done += 1

    return list(S)
