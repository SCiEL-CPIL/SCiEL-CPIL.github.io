import time
from itertools import combinations
import random

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
    # Cases: 2 from selected\{a} and x.
    for a in selected:
        others_no_a = [v for v in selected if v != a]
        if len(others_no_a) >= 2:
            for pair in combinations(others_no_a, 2):
                s = pair[0] + pair[1] + x
                if s % a == 0:
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

def _greedy_removal(pool, target_size=48, time_limit=5.0):
    """
    Start with a pool, remove elements causing violations until valid or target size reached.
    """
    selected = list(pool)
    start_time = time.time()
    
    while len(selected) > target_size:
        if time.time() - start_time > time_limit:
            break
            
        # Count violations
        violation_count = {x: 0 for x in selected}
        found_violation = False
        
        # Check all a
        for a in selected:
            others = [v for v in selected if v != a]
            if len(others) < 3:
                continue
            for t in combinations(others, 3):
                s = sum(t)
                if s % a == 0:
                    found_violation = True
                    violation_count[a] += 1
                    violation_count[t[0]] += 1
                    violation_count[t[1]] += 1
                    violation_count[t[2]] += 1
        
        if not found_violation:
            # Valid set, but larger than target
            # Remove smallest elements to reduce size?
            # Or just stop? The task says "at most 48".
            # If we have 49 valid, we should return 48.
            # Remove the "least critical" element.
            # Heuristic: Remove the smallest one (less likely to be needed for divisibility safety of others? No, small ones are dangerous divisors. Keeping them is risky. Removing them is safe.)
            # So remove smallest.
            selected.sort()
            selected.pop(0)
            continue
            
        # Remove element with max violation count
        # Tie-breaker: remove smallest value (dangerous divisor)
        # We want to keep large numbers.
        worst_elem = min(selected, key=lambda x: (-violation_count[x], x))
        selected.remove(worst_elem)
        
    # If we exited because time limit or no violations, check if we need to trim
    if len(selected) > target_size:
        selected.sort()
        # Remove smallest to be safe?
        # Actually, if it's valid, any subset is valid.
        # So we can just take the first 48.
        # But which 48?
        # Large numbers are "safer" in the sense that they don't divide sums of smaller numbers easily?
        # No, if a is large, it's safe.
        # So keep the largest 48.
        selected.sort(reverse=True)
        selected = selected[:target_size]
        
    return selected

def _local_search(selected, time_limit):
    """Local search with swap and add moves."""
    start = time.time()
    best = list(selected)
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, 193) if x not in best]
        
        # Try adding
        for x in candidates:
            if time.time() - start > time_limit:
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add one
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
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit
    
    best_result = []
    
    # Strategy 1: Greedy Removal from high-range pools
    # Pool 144-192 (49 elements)
    r1 = _greedy_removal(range(144, 193), 48, time_limit=2.0)
    if len(r1) > len(best_result):
        best_result = r1
        
    # Pool 130-192 (63 elements)
    if time.time() < deadline - 10:
        r2 = _greedy_removal(range(130, 193), 48, time_limit=3.0)
        if len(r2) > len(best_result):
            best_result = r2
            
    # Pool 100-192 (93 elements) - might be slow
    if time.time() < deadline - 15:
        r3 = _greedy_removal(range(100, 193), 48, time_limit=5.0)
        if len(r3) > len(best_result):
            best_result = r3

    # Strategy 2: Greedy Addition (Large to Small)
    if time.time() < deadline - 5:
        order1 = list(range(192, 0, -1))
        r4 = _greedy_build(order1)
        if len(r4) > len(best_result):
            best_result = r4
            
    # Strategy 3: Randomized Greedy
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        order = list(range(1, 193))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r
            
    # Strategy 4: Local Search on best result
    if time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 1
        best_result = _local_search(best_result, remaining_time)
        
    return best_result
