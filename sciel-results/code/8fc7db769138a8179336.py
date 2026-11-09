import time
import random
from itertools import combinations

def _is_valid_set(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    m = len(selected)
    if m < 4:
        return True
    for a in selected:
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    m = len(selected)
    # Check: does x divide sum of any 3 from selected?
    if m >= 3:
        for t in combinations(selected, 3):
            if sum(t) % x == 0:
                return False
    # Check: for each a in selected, does a divide sum of any 3 from (selected ∪ {x}) \ {a}?
    for a in selected:
        others_no_a = [v for v in selected if v != a]
        if len(others_no_a) >= 2:
            for pair in combinations(others_no_a, 2):
                s = pair[0] + pair[1] + x
                if s % a == 0:
                    return False
    return True

def _can_add_two(x1, x2, selected):
    """Check if both x1 and x2 can be added to selected without violating constraints."""
    if not _can_add(x1, selected):
        return False
    if not _can_add(x2, selected + [x1]):
        return False
    return True

def _greedy_build(order):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= 48:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit):
    """Local search with 1-add, 1-swap, and 2-swap moves."""
    start = time.time()
    best = list(selected)
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, 193) if x not in best]
        
        # Try 1-add
        for x in candidates:
            if time.time() - start > time_limit:
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        
        if not improved:
            # Try 1-swap: remove one, add one
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                temp = best[:i] + best[i+1:]
                for x in candidates:
                    if _can_add(x, temp):
                        best = temp + [x]
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try 2-swap: remove two, add two
            n = len(best)
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                for j in range(i+1, n):
                    if time.time() - start > time_limit:
                        break
                    temp = best[:i] + best[i+1:j] + best[j+1:]
                    found = False
                    for ci in range(len(candidates)):
                        if time.time() - start > time_limit:
                            break
                        x1 = candidates[ci]
                        for cj in range(ci+1, len(candidates)):
                            if time.time() - start > time_limit:
                                break
                            x2 = candidates[cj]
                            if _can_add_two(x1, x2, temp):
                                best = temp + [x1, x2]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if improved:
                    break
        
        if not improved:
            # Try remove-one-then-add-two
            if len(best) > 10:
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        break
                    temp = best[:i] + best[i+1:]
                    found = False
                    for ci in range(len(candidates)):
                        if time.time() - start > time_limit:
                            break
                        x1 = candidates[ci]
                        for cj in range(ci+1, len(candidates)):
                            if time.time() - start > time_limit:
                                break
                            x2 = candidates[cj]
                            if _can_add_two(x1, x2, temp):
                                best = temp + [x1, x2]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        
        if not improved:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin
    
    best_result = []
    
    # Strategy 1: Greedy from large to small
    order1 = list(range(192, 0, -1))
    r1 = _greedy_build(order1)
    best_result = r1
    
    # Strategy 2: Greedy from small to large
    order2 = list(range(1, 193))
    r2 = _greedy_build(order2)
    if len(r2) > len(best_result):
        best_result = r2
    
    # Strategy 3: Randomized greedy with more seeds
    for seed in range(150):
        if time.time() > deadline - 30:
            break
        random.seed(seed)
        order = list(range(1, 193))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r
    
    # Strategy 4: Start with high-range numbers
    for hi_start in [192, 180, 168, 156, 144, 132, 120, 108]:
        if time.time() > deadline - 30:
            break
        order = list(range(hi_start, 0, -1))
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r
    
    # Local search on best result
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        best_result = _local_search(best_result, remaining_time)
    
    # Try local search on other good results too
    if time.time() < deadline - 3:
        for seed in range(30):
            if time.time() > deadline - 3:
                break
            random.seed(1000 + seed)
            order = list(range(1, 193))
            random.shuffle(order)
            r = _greedy_build(order)
            if len(r) >= len(best_result) - 2:
                remaining_time = min(deadline - time.time() - 1, 5.0)
                if remaining_time > 0.5:
                    r = _local_search(r, remaining_time)
                    if len(r) > len(best_result):
                        best_result = r
    
    return best_result
