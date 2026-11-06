import random
import time
from itertools import combinations

def _can_add(S, x):
    """
    Check if x can be added to S.
    S is a list of distinct integers.
    """
    if x in S:
        return False
        
    # 1. Check if x divides sum of any 3 distinct elements in S
    if len(S) >= 3:
        for b, c, d in combinations(S, 3):
            if (b + c + d) % x == 0:
                return False
            
    # 2. Check if any a in S divides sum of x and any 2 distinct elements in S
    for a in S:
        target = (-x) % a
        # We need distinct b, c in S\{a} such that (b+c) % a == target
        # Count remainders of S\{a} mod a
        counts = [0] * a
        for v in S:
            if v != a:
                counts[v % a] += 1
                
        for v in S:
            if v == a:
                continue
            r1 = v % a
            r2 = (target - r1) % a
            if r1 == r2:
                if counts[r1] >= 2:
                    return False
            else:
                if counts[r1] >= 1 and counts[r2] >= 1:
                    return False
                    
    return True

def _local_search(initial_set, max_iterations=100):
    S = list(initial_set)
    all_candidates = set(range(1, 193))
    
    for _ in range(max_iterations):
        changed = False
        
        # 1. Try to add
        remaining = [x for x in all_candidates if x not in S]
        random.shuffle(remaining)
        
        for x in remaining:
            if _can_add(S, x):
                S.append(x)
                changed = True
                break
        
        if changed:
            continue
        
        # 2. Try to swap
        if not S:
            break
            
        a = random.choice(S)
        remaining = [x for x in all_candidates if x not in S]
        if not remaining:
            break
            
        random.shuffle(remaining)
        
        swapped = False
        for _ in range(10):
            b = random.choice(remaining)
            S_temp = [x for x in S if x != a]
            if _can_add(S_temp, b):
                S = S_temp
                S.append(b)
                swapped = True
                break
        
        if swapped:
            continue
            
        # 3. Stuck
        break
        
    return S

def construct():
    """
    Returns a list of distinct positive integers <= 192, maximizing cardinality,
    such that no selected integer divides the sum of any 3 distinct OTHER selected integers.
    """
    start_time = time.time()
    timeout = 55.0
    best_set = []
    
    for i in range(20):
        if time.time() - start_time > timeout:
            break
            
        S = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        for x in candidates:
            if _can_add(S, x):
                S.append(x)
            if time.time() - start_time > timeout:
                break
                
        S = _local_search(S, max_iterations=50)
        
        if len(S) > len(best_set):
            best_set = S
            
        if len(best_set) >= 48:
            return best_set
            
    return best_set
