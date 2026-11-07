"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 256
CANDIDATES = list(range(2, N + 1))

def _can_add(a, S):
    """Check if a can be added to set S without violating the condition."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            # Check all three roles for the divisor
            if (b * c) % a == 0:
                return False
            if (a * c) % b == 0:
                return False
            if (a * b) % c == 0:
                return False
    return True

def _is_valid_set(S):
    """Verify the entire set satisfies the condition."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_desc():
    """Greedy: add numbers in descending order."""
    S = []
    for a in reversed(CANDIDATES):
        if _can_add(a, S):
            S.append(a)
    return S

def _greedy_asc():
    """Greedy: add numbers in ascending order."""
    S = []
    for a in CANDIDATES:
        if _can_add(a, S):
            S.append(a)
    return S

def _greedy_random(seed_offset=0):
    """Greedy with randomized order."""
    rng = random.Random(seed_offset)
    candidates = CANDIDATES[:]
    rng.shuffle(candidates)
    S = []
    for a in candidates:
        if _can_add(a, S):
            S.append(a)
    return S

def _local_search(S, time_limit=40.0):
    """
    Local search: try 1-for-1 swaps to improve the set.
    S is a list. Returns the best set found.
    """
    start_time = time.time()
    S_set = set(S)
    best_S = S_set.copy()
    best_size = len(S_set)
    
    def check_swap(x, y, S_list):
        """Check if removing x and adding y keeps the set valid.
        Only need to check triples involving y (since removing x only removes constraints)."""
        S_minus_x = [v for v in S_list if v != x]
        n = len(S_minus_x)
        for i in range(n):
            b = S_minus_x[i]
            for j in range(n):
                if i == j:
                    continue
                c = S_minus_x[j]
                if (b * c) % y == 0:
                    return False
                if (y * c) % b == 0:
                    return False
                if (y * b) % c == 0:
                    return False
        return True
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        S_list = list(S_set)
        # Shuffle to avoid getting stuck
        random.shuffle(S_list)
        for x in S_list:
            if time.time() - start_time >= time_limit:
                break
            # Try to find a y to swap in
            y_candidates = [y for y in CANDIDATES if y not in S_set]
            random.shuffle(y_candidates)
            for y in y_candidates:
                if check_swap(x, y, S_list):
                    S_set.remove(x)
                    S_set.add(y)
                    improved = True
                    if len(S_set) > best_size:
                        best_S = S_set.copy()
                        best_size = len(S_set)
                    break
            if improved:
                break
    
    return list(best_S)

def _two_for_one_search(S, time_limit=30.0):
    """
    Try removing two elements and adding one or two others to potentially gain size.
    This is more expensive but can escape 1-for-1 local optima.
    """
    start_time = time.time()
    S_set = set(S)
    best_S = S_set.copy()
    best_size = len(S_set)
    
    S_list = list(S_set)
    n = len(S_list)
    
    # Try removing two, adding two (net zero) or removing two, adding three (net +1)
    # Let's try removing one, adding two (net +1)
    for i in range(n):
        if time.time() - start_time >= time_limit:
            break
        x = S_list[i]
        S_minus_x = [v for v in S_list if v != x]
        
        # Try adding two elements y, z to S_minus_x
        y_candidates = [y for y in CANDIDATES if y not in S_set]
        random.shuffle(y_candidates)
        # Limit candidates for speed
        y_candidates = y_candidates[:100]
        
        for y in y_candidates:
            if time.time() - start_time >= time_limit:
                break
            # Check if y can be added to S_minus_x
            if not _can_add(y, S_minus_x):
                continue
            S_temp = S_minus_x + [y]
            
            # Now try to add z
            z_candidates = [z for z in CANDIDATES if z not in S_set and z != y]
            random.shuffle(z_candidates)
            z_candidates = z_candidates[:50]
            
            for z in z_candidates:
                if _can_add(z, S_temp):
                    # Success: removed x, added y and z. Net +1
                    new_S = set(S_temp + [z])
                    if len(new_S) > best_size:
                        best_S = new_S
                        best_size = len(new_S)
                        S_set = new_S
                        S_list = list(S_set)
                        n = len(S_list)
                    break
    
    return list(best_S)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Try multiple greedy strategies
    best_result = []
    best_size = 0
    
    # Descending greedy (large numbers first - likely best)
    if time.time() - start_time < deadline:
        S = _greedy_desc()
        if len(S) > best_size:
            best_result = S
            best_size = len(S)
    
    # Ascending greedy
    if time.time() - start_time < deadline:
        S = _greedy_asc()
        if len(S) > best_size:
            best_result = S
            best_size = len(S)
    
    # Several random greedy restarts
    for seed in range(20):
        if time.time() - start_time >= deadline:
            break
        S = _greedy_random(seed)
        if len(S) > best_size:
            best_result = S
            best_size = len(S)
    
    # Local search on the best result
    if time.time() - start_time < deadline:
        remaining = deadline - (time.time() - start_time)
        S = _local_search(best_result, time_limit=remaining * 0.6)
        if len(S) > best_size:
            best_result = S
            best_size = len(S)
    
    # Two-for-one search
    if time.time() - start_time < deadline:
        remaining = deadline - (time.time() - start_time)
        S = _two_for_one_search(best_result, time_limit=remaining * 0.5)
        if len(S) > best_size:
            best_result = S
            best_size = len(S)
    
    # Final local search
    if time.time() - start_time < deadline:
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            S = _local_search(best_result, time_limit=remaining * 0.8)
            if len(S) > best_size:
                best_result = S
                best_size = len(S)
    
    # Verify
    if _is_valid_set(best_result):
        return list(best_result)
    else:
        # Fallback: just primes
        from math import isqrt
        return [a for a in range(2, N+1) if all(a % d for d in range(2, isqrt(a)+1))]

# EVOLVE-BLOCK-END
