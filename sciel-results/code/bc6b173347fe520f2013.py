import time
from itertools import combinations
import random

def _is_valid_set_fast(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    m = len(selected)
    if m < 4:
        return True
    sel = list(selected)
    for i, a in enumerate(sel):
        others = sel[:i] + sel[i+1:]
        for j in range(len(others)):
            for k in range(j+1, len(others)):
                for l in range(k+1, len(others)):
                    if (others[j] + others[k] + others[l]) % a == 0:
                        return False
    return True

def _can_add_fast(x, selected):
    """Check if x can be added to selected without violating constraints."""
    m = len(selected)
    # Check: does x divide sum of any 3 from selected?
    if m >= 3:
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    # Check: for each a in selected, does a divide sum of any 3 from (selected ∪ {x}) \ {a}?
    # Only need to check combinations that include x: 2 from selected\{a} and x.
    for i, a in enumerate(selected):
        others_no_a = selected[:i] + selected[i+1:]
        if len(others_no_a) >= 2:
            for j in range(len(others_no_a)):
                for k in range(j+1, len(others_no_a)):
                    if (others_no_a[j] + others_no_a[k] + x) % a == 0:
                        return False
    return True

def _greedy_build(order):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= 48:
            break
        if _can_add_fast(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit):
    """Local search with swap and add moves."""
    start = time.time()
    best = list(selected)
    best_len = len(best)
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, 193) if x not in best]
        
        # Try adding
        for x in candidates:
            if time.time() - start > time_limit:
                break
            if _can_add_fast(x, best):
                best.append(x)
                best_len = len(best)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add one
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                temp = best[:i] + best[i+1:]
                for x in candidates:
                    if _can_add_fast(x, temp):
                        best = temp + [x]
                        best_len = len(best)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit
    
    best_result = []
    
    # Strategy 1: Greedy Addition (Large to Small) - large numbers are safer
    order1 = list(range(192, 0, -1))
    r1 = _greedy_build(order1)
    if len(r1) > len(best_result):
        best_result = r1
        
    # Strategy 2: Randomized Greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        random.seed(seed)
        order = list(range(1, 193))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r
            
    # Strategy 3: Local Search on best result
    if time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 1
        best_result = _local_search(best_result, remaining_time)
        
    return best_result
