"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
from itertools import combinations_with_replacement
from bisect import bisect_left, insort

def _triple_sums_sorted(arr):
    """Compute all unordered triple sums (with repetition) for sorted arr, return sorted list."""
    n = len(arr)
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(arr[i] + arr[j] + arr[k])
    sums.sort()
    return sums

def _has_collision_sorted(arr):
    """Check if sorted arr has any duplicate triple sums."""
    sums = _triple_sums_sorted(arr)
    for i in range(1, len(sums)):
        if sums[i] == sums[i-1]:
            return True
    return False

def _new_sums_with_x(x, arr):
    """Compute triple sums involving x, given sorted arr (not including x)."""
    n = len(arr)
    sums = [3 * x]
    for a in arr:
        sums.append(2 * x + a)
        sums.append(x + 2 * a)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + arr[i] + arr[j])
    return sums

def _is_valid_with_x(x, arr, existing_sums_set):
    """Check if adding x to arr creates no collision."""
    new_sums = _new_sums_with_x(x, arr)
    for s in new_sums:
        if s in existing_sums_set:
            return False
    return True

def _compute_sums_set(arr):
    """Compute the set of all triple sums for arr."""
    sums = set()
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(arr[i] + arr[j] + arr[k])
    return sums

def _greedy_powers_of_3(max_val=4096):
    """Construct using powers of 3 type approach."""
    selected = []
    sums_set = set()
    
    # Start with 1, then try to add next power of 3 or similar
    # Actually, let's use a greedy approach that picks the smallest valid element
    for x in range(1, max_val + 1):
        if _is_valid_with_x(x, selected, sums_set):
            selected.append(x)
            new_sums = _new_sums_with_x(x, selected[:-1])
            for s in new_sums:
                sums_set.add(s)
    
    return selected, sums_set

def _greedy_with_backtrack(max_val=4096, time_limit=30, max_size_target=80):
    """Greedy with backtracking to find larger sets."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    best = []
    best_sums = set()
    
    def backtrack(current, sums_set, depth):
        nonlocal best, best_sums
        if time.time() > deadline:
            return
        
        if len(current) > len(best):
            best = list(current)
            best_sums = set(sums_set)
        
        if len(current) >= max_size_target:
            return
        
        # Try adding next candidate
        # We'll try candidates from 1 to max_val
        # To be efficient, we'll try in order and use pruning
        for x in range(1, max_val + 1):
            if time.time() > deadline:
                return
            if x in current:
                continue
            if _is_valid_with_x(x, current, sums_set):
                current.append(x)
                new_sums = _new_sums_with_x(x, current[:-1])
                for s in new_sums:
                    sums_set.add(s)
                backtrack(current, sums_set, depth + 1)
                current.pop()
                for s in new_sums:
                    sums_set.discard(s)
                # Don't break - try other candidates too for better search
                # But to limit branching, break after first success
                break
    
    backtrack([], set(), 0)
    return best, best_sums

def _smart_greedy(max_val=4096, time_limit=40):
    """
    Smart greedy: start with a good initial set, then try to fill gaps.
    Use multiple strategies and keep the best.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best = []
    best_sums = set()
    best_size = 0
    
    def update_best(candidate, sums_set):
        nonlocal best, best_sums, best_size
        if len(candidate) > best_size:
            best = list(candidate)
            best_sums = set(sums_set)
            best_size = len(candidate)
    
    # Strategy 1: Pure greedy from 1
    selected = []
    sums_set = set()
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
        if _is_valid_with_x(x, selected, sums_set):
            selected.append(x)
            new_sums = _new_sums_with_x(x, selected[:-1])
            for s in new_sums:
                sums_set.add(s)
    update_best(selected, sums_set)
    
    if time.time() > deadline:
        return best, best_sums
    
    # Strategy 2: Try skipping small numbers to allow larger ones
    # Start with a gap, then fill
    for skip_start in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]:
        if time.time() > deadline:
            break
        selected = []
        sums_set = set()
        for x in range(skip_start, max_val + 1):
            if time.time() > deadline:
                break
            if _is_valid_with_x(x, selected, sums_set):
                selected.append(x)
                new_sums = _new_sums_with_x(x, selected[:-1])
                for s in new_sums:
                    sums_set.add(s)
        update_best(selected, sums_set)
    
    if time.time() > deadline:
        return best, best_sums
    
    # Strategy 3: Try to add elements to the best set found
    current = list(best)
    current_sums = set(best_sums)
    current_set = set(current)
    
    # Try adding any element not in current
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if _is_valid_with_x(x, current, current_sums):
            current.append(x)
            current_set.add(x)
            new_sums = _new_sums_with_x(x, current[:-1])
            for s in new_sums:
                current_sums.add(s)
    
    update_best(current, current_sums)
    
    if time.time() > deadline:
        return best, best_sums
    
    # Strategy 4: Local search - try swapping elements
    # For each element in best, try replacing it with a different element
    for _ in range(200):  # Limit iterations
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            
            old = best[i]
            # Try replacing old with some other value
            # We'll try a few candidates
            for x in range(1, max_val + 1):
                if time.time() > deadline:
                    break
                if x == old or x in best:
                    continue
                
                # Check if replacing old with x is valid
                # Remove old, add x
                temp = [v for v in best if v != old]
                # This is inefficient; let's use a different approach
                
                # Actually, let's just try to add x to best without old
                # First, compute sums of best without old
                # This is complex; skip for now
                pass
        
        if not improved:
            break
    
    return best, best_sums

def _backtrack_search(max_val=4096, time_limit=50, target_size=80):
    """
    Backtracking search with pruning.
    Try to build a set of size target_size.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best = []
    best_sums = set()
    best_size = 0
    
    def update_best(candidate, sums_set):
        nonlocal best, best_sums, best_size
        if len(candidate) > best_size:
            best = list(candidate)
            best_sums = set(sums_set)
            best_size = len(candidate)
    
    def backtrack(current, sums_set, start_candidate):
        nonlocal best, best_sums, best_size
        if time.time() > deadline:
            return
        
        if len(current) > best_size:
            best = list(current)
            best_sums = set(sums_set)
            best_size = len(current)
        
        if len(current) >= target_size:
            return
        
        # Try candidates from start_candidate to max_val
        for x in range(start_candidate, max_val + 1):
            if time.time() > deadline:
                return
            
            if _is_valid_with_x(x, current, sums_set):
                current.append(x)
                new_sums = _new_sums_with_x(x, current[:-1])
                for s in new_sums:
                    sums_set.add(s)
                
                backtrack(current, sums_set, x + 1)
                
                current.pop()
                for s in new_sums:
                    sums_set.discard(s)
    
    backtrack([], set(), 1)
    return best, best_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_result = []
    best_sums = set()
    best_size = 0
    
    def update_best(candidate, sums_set):
        nonlocal best_result, best_sums, best_size
        if len(candidate) > best_size:
            best_result = list(candidate)
            best_sums = set(sums_set)
            best_size = len(candidate)
    
    # Phase 1: Quick greedy to get a baseline
    if time.time() < deadline - 10:
        selected, sums_set = _greedy_powers_of_3(4096)
        update_best(selected, sums_set)
    
    if time.time() > deadline:
        return best_result
    
    # Phase 2: Smart greedy with multiple strategies
    if time.time() < deadline - 20:
        selected, sums_set = _smart_greedy(4096, time_limit=min(20, deadline - time.time()))
        update_best(selected, sums_set)
    
    if time.time() > deadline:
        return best_result
    
    # Phase 3: Backtracking search with remaining time
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        selected, sums_set = _backtrack_search(4096, time_limit=remaining_time, target_size=80)
        update_best(selected, sums_set)
    
    return best_result
