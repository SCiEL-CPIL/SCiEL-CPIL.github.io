"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
from itertools import combinations_with_replacement

def _new_sums_involving(x, S):
    """Compute all triple sums that include x, given existing sorted set S."""
    n = len(S)
    sums = [3 * x]
    for s in S:
        sums.append(2 * x + s)
        sums.append(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + S[i] + S[j])
    return sums

def _check_collision(new_sums, existing_sums_set):
    """Check if any new sum collides with existing sums."""
    for s in new_sums:
        if s in existing_sums_set:
            return True
    return False

def _greedy_build(start=1, end=4096, step=1):
    """Greedy construction scanning from start to end with given step."""
    selected = []
    existing_sums_set = set()
    
    for x in range(start, end + 1, step):
        new_sums = _new_sums_involving(x, selected)
        if not _check_collision(new_sums, existing_sums_set):
            selected.append(x)
            for s in new_sums:
                existing_sums_set.add(s)
    
    return selected, existing_sums_set

def _local_search_improve(selected, existing_sums_set, time_limit=30):
    """Try to improve the set via local search: swap elements."""
    if time.time() > time_limit:
        return selected
    
    best = list(selected)
    best_sums = set(existing_sums_set)
    best_size = len(best)
    
    # Try swapping elements
    for _ in range(5000):
        if time.time() > time_limit:
            break
        
        if len(best) >= 80:
            break
        
        # Try adding a new element
        found = False
        for x in range(1, 4097):
            if x in best:
                continue
            new_sums = _new_sums_involving(x, best)
            if not _check_collision(new_sums, best_sums):
                best.append(x)
                for s in new_sums:
                    best_sums.add(s)
                best_size = len(best)
                found = True
                break
        
        if found:
            continue
        
        # Try swapping: remove one element, add another
        if best_size < 80:
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                removed = best[i]
                # Temporarily remove
                new_best = best[:i] + best[i+1:]
                new_sums_set = set(best_sums)
                # Remove sums involving removed element
                # This is complex; let's just try a different approach
                # Try adding elements we haven't tried yet
                pass
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Linear greedy from 1
    selected, sums_set = _greedy_build(1, 4096, 1)
    if len(selected) > best_size:
        best_size = len(selected)
        best_result = list(selected)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Try different starting points and steps
    for start in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]:
        if time.time() > deadline:
            break
        for step in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
            if time.time() > deadline:
                break
            selected, sums_set = _greedy_build(start, 4096, step)
            if len(selected) > best_size:
                best_size = len(selected)
                best_result = list(selected)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 3: Try to add more elements to the best result
    # We compute the full set of triple sums for the current best_result
    # and then greedily try to add any other integers from 1..4096.
    current = list(best_result)
    
    # Compute all triple sums for 'current'
    current_sums = set()
    n_curr = len(current)
    for i in range(n_curr):
        for j in range(i, n_curr):
            for k in range(j, n_curr):
                current_sums.add(current[i] + current[j] + current[k])
    
    # Try to add elements
    # We iterate through all possible candidates. 
    # Note: 'current' is not sorted necessarily, but _new_sums_involving works regardless.
    # However, for 'x in current' to be efficient, we could use a set, but N is small.
    current_set = set(current)
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        
        new_sums = _new_sums_involving(x, current)
        
        # Check for collision with existing sums
        has_collision = False
        for s in new_sums:
            if s in current_sums:
                has_collision = True
                break
        
        if not has_collision:
            current.append(x)
            current_set.add(x)
            for s in new_sums:
                current_sums.add(s)
            best_size = len(current)
    
    best_result = current
    return best_result
