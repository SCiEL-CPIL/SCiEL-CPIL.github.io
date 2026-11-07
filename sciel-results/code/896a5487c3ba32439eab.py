import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid(s):
    """
    Check if set s violates the condition:
    No three distinct members a, b, c in s such that a+b, a+c, b+c are all in s.
    Note: a+b, a+c, b+c don't need to be distinct from each other or from a,b,c.
    """
    s_list = sorted(s)
    s_set = set(s_list)
    n = len(s_list)
    
    # For each triple of distinct elements (a, b, c) with a < b < c:
    # Check if a+b in s, a+c in s, b+c in s
    for i in range(n):
        a = s_list[i]
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab not in s_set:
                continue
            for k in range(j+1, n):
                c = s_list[k]
                ac = a + c
                bc = b + c
                if ac in s_set and bc in s_set:
                    return False
    return True

def _is_valid_fast(s):
    """
    Faster validity check using bitmasks or optimized loops.
    For sets up to ~128 elements in range 1..128, this is fast enough.
    """
    s_set = set(s)
    s_list = sorted(s)
    n = len(s_list)
    
    # Precompute pair sums
    pair_sums = {}
    for i in range(n):
        for j in range(i+1, n):
            ps = s_list[i] + s_list[j]
            if ps <= 256:  # max possible sum is 128+128=256
                if ps not in pair_sums:
                    pair_sums[ps] = []
                pair_sums[ps].append((i, j))
    
    # For each triple (a, b, c) with a < b < c, check if all three pair sums are in s
    for i in range(n):
        a = s_list[i]
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab not in s_set:
                continue
            for k in range(j+1, n):
                c = s_list[k]
                ac = a + c
                bc = b + c
                if ac in s_set and bc in s_set:
                    return False
    return True

def _greedy_construct(deadline):
    """
    Greedy construction starting from upper half, then try to add/remove elements.
    """
    best = set(range(65, 129))  # Start with 64 elements: 65..128
    
    # Try to add elements from 1..64
    for v in range(1, 65):
        if time.time() > deadline:
            break
        trial = best | {v}
        if _is_valid(trial):
            best = trial
    
    # Local search: try to swap elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and see if we can add two new ones
        for rem in sorted(best):
            if time.time() > deadline:
                break
            trial_base = best - {rem}
            # Try to add elements from 1..128 not in trial_base
            added = 0
            for add in range(1, 129):
                if add in trial_base or add == rem:
                    continue
                if time.time() > deadline:
                    break
                trial = trial_base | {add}
                if _is_valid(trial):
                    trial_base = trial
                    added += 1
            if len(trial_base) > len(best):
                best = trial_base
                improved = True
    
    return best

def _randomized_local_search(deadline, seed=42):
    """
    Randomized local search with restarts.
    """
    random.seed(seed)
    best = set(range(65, 129))
    best_size = len(best)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start from a random valid set near the best
        current = set(best)
        # Perturb: remove a random few and add random few
        for _ in range(3):
            if time.time() > deadline:
                break
            rem = random.choice(list(current))
            current = current - {rem}
            # Try to add a random element
            candidates = [v for v in range(1, 129) if v not in current]
            if candidates:
                add = random.choice(candidates)
                trial = current | {add}
                if _is_valid(trial):
                    current = trial
        
        # Local search on current
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each element
            for add in range(1, 129):
                if add in current:
                    continue
                if time.time() > deadline:
                    break
                trial = current | {add}
                if _is_valid(trial):
                    current = trial
                    improved = True
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
            
            # Try removing one and adding two
            for rem in list(current):
                if time.time() > deadline:
                    break
                trial_base = current - {rem}
                # Try to add elements
                added = 0
                for add in range(1, 129):
                    if add in trial_base or add == rem:
                        continue
                    if time.time() > deadline:
                        break
                    trial = trial_base | {add}
                    if _is_valid(trial):
                        trial_base = trial
                        added += 1
                if len(trial_base) > len(current):
                    current = trial_base
                    improved = True
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin (90s limit)
    
    # Start with the seed solution
    best = set(range(65, 129))
    best_size = 64
    
    # Step 1: Greedy construction
    if time.time() < deadline - 20:
        greedy_result = _greedy_construct(deadline - 20)
        if len(greedy_result) > best_size:
            best = greedy_result
            best_size = len(greedy_result)
    
    # Step 2: Randomized local search
    if time.time() < deadline - 5:
        rls_result = _randomized_local_search(deadline - 5)
        if len(rls_result) > best_size:
            best = rls_result
            best_size = len(rls_result)
    
    # Final check: make sure the result is valid
    if not _is_valid(best):
        # Fallback to seed
        best = set(range(65, 129))
    
    # Return as list of Python ints
    return [int(x) for x in sorted(best)]

# EVOLVE-BLOCK-END
