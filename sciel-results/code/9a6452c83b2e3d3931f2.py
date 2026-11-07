import time
import random

def _is_valid_range(s):
    """
    Check if sorted s satisfies Max_k < Min_{k+1} for all k.
    """
    n = len(s)
    if n < 2:
        return True
    s_sorted = sorted(s)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + s_sorted[i]
        
    for k in range(1, n):
        max_k = prefix[n] - prefix[n - k]
        min_k1 = prefix[k + 1]
        if max_k >= min_k1:
            return False
    return True

def _is_valid_exact(s):
    """
    Exact check using bitsets.
    """
    n = len(s)
    if n < 2:
        return True
    
    sums = [0] * (n + 1)
    sums[0] = 1
    
    for x in s:
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << x)
            
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    # Phase 1: Find largest Arithmetic Progression satisfying Range Condition
    # This is very fast and likely finds a large set.
    for n in range(60, 0, -1):
        if time.time() > deadline - 5.0:
            break
        for a in range(1, 257 - n + 1):
            s = list(range(a, a + n))
            if _is_valid_range(s):
                if len(s) > len(best_set):
                    best_set = s
                # Found a valid AP of size n. Since we iterate n down, 
                # the first one is the largest AP.
                # But maybe a non-AP of same size or larger exists?
                # We can break here to use time for Phase 2?
                # Or keep searching for larger n?
                # We iterate n from 60 down. So first hit is max n for AP.
                break
        
    # Phase 2: Local Search / Simulated Annealing to improve or find non-AP solutions
    # Start with best_set found in Phase 1.
    # If best_set is empty, start with a small valid set like [1, 2].
    if not best_set:
        best_set = [1, 2]
        
    current_set = best_set[:]
    current_size = len(current_set)
    
    # Use Exact Check for Phase 2 to explore the "loose" space.
    # We want to maximize size.
    
    # Helper to check if we can add x
    def can_add(s, x):
        if x in s:
            return False
        return _is_valid_exact(s + [x])
    
    # Simple Hill Climbing with Restarts
    # Since Exact Check is fast, we can do many steps.
    
    for restart in range(100):
        if time.time() > deadline - 2.0:
            break
            
        # Start from current best or a random valid set
        if restart == 0:
            s = current_set[:]
        else:
            # Generate a random dense cluster
            n_try = random.randint(10, 40)
            a_try = random.randint(1, max(1, 256 - n_try))
            s = list(range(a_try, a_try + n_try))
            if not _is_valid_exact(s):
                 # Fallback to small set
                 s = [1, 2]
            
        s_set = set(s)
        
        # Try to add elements
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try all unused elements? 256 is small.
            for x in range(1, 257):
                if x in s_set:
                    continue
                if _is_valid_exact(s + [x]):
                    s.append(x)
                    s_set.add(x)
                    improved = True
                    if len(s) > len(best_set):
                        best_set = s[:]
                    # Restart the while loop to check again
                    break 
            
        # Try swaps to free up space?
        # If we can't add, maybe swap a large element with a smaller one to allow more adds?
        # Or swap to make the set "tighter"?
        
        # Simple Swap Move
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            if len(s) < 2:
                break
            i = random.randint(0, len(s) - 1)
            # Pick a candidate to swap with
            candidates = [x for x in range(1, 257) if x not in s_set]
            if not candidates:
                break
            j = random.choice(candidates)
            
            new_s = s[:]
            new_s[i] = j
            new_s_set = set(new_s)
            if len(new_s_set) != len(new_s):
                continue
            
            if _is_valid_exact(new_s):
                s = new_s
                s_set = new_s_set
                if len(s) > len(best_set):
                    best_set = s[:]
                
    return list(best_set)
