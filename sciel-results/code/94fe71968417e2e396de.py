import random
import numpy as np

N = 256

def _is_sidon(s):
    """Check if set s is a Sidon set (all pairwise sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _is_maximal(s, n):
    """Check if Sidon set s is inclusion-maximal in [1, n]."""
    s_set = set(s)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.add(s[i] + s[j])
    for x in range(1, n + 1):
        if x in s_set:
            continue
        # Check if adding x would break Sidon property
        new_sums = set()
        for a in s:
            new_sums.add(x + a)
        new_sums.add(2 * x)
        if sums.intersection(new_sums) or len(new_sums) != len(new_sums):
            # x cannot be added, so maximality holds for this x
            continue
        else:
            # x CAN be added, so s is not maximal
            return False
    return True

def _can_add(s, sums, x):
    """Check if x can be added to s given its current sums."""
    for a in s:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True

def construct():
    # Greedy Maximal Coverage Strategy
    S = []
    S_set = set()
    sums = set()
    
    all_elements = list(range(1, N + 1))
    
    while True:
        # Find all candidates that can be added to current S
        candidates = []
        for x in all_elements:
            if x in S_set:
                continue
            if _can_add(S, sums, x):
                candidates.append(x)
        
        if not candidates:
            break
        
        best_x = None
        max_blocked = -1
        
        for x in candidates:
            # Simulate adding x
            # New sums to add: {x+a for a in S} U {2x}
            new_sums_to_add = set([x + a for a in S] + [2 * x])
            
            # Count how many other candidates y become non-addable
            blocked_count = 0
            for y in candidates:
                if y == x:
                    continue
                # y is blocked if y+x or 2y is in new_sums_to_add
                if (y + x) in new_sums_to_add or (2 * y) in new_sums_to_add:
                    blocked_count += 1
            
            if blocked_count > max_blocked:
                max_blocked = blocked_count
                best_x = x
        
        # Add the best candidate
        S.append(best_x)
        S_set.add(best_x)
        for a in S[:-1]:
            sums.add(best_x + a)
        sums.add(2 * best_x)
        
    return S
