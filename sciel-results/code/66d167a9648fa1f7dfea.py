"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills at 90s wall-clock. Use 85s safety margin.
_DEADLINE = 85.0

def _is_valid_full(selected):
    """Check that no element divides the sum of any 3 distinct OTHER elements."""
    n = len(selected)
    if n < 4:
        return True
    sel = list(selected)
    for i, a in enumerate(sel):
        others = [v for j, v in enumerate(sel) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    # 1) x as the tested element: sum of 3 others divisible by x
    if len(selected) >= 3:
        for t in combinations(selected, 3):
            if sum(t) % x == 0:
                return False
                
    # 2) Each existing element a: check if x participates in a 3-sum divisible by a
    #    We need b, c from selected (b != c, b != a, c != a) such that (x + b + c) % a == 0
    for a in selected:
        # Iterate pairs from selected
        # Optimization: if x >= a, x % a is fixed.
        # We need (b + c) % a == (-x) % a
        target = (-x) % a
        found = False
        # Simple pair iteration
        sel_list = list(selected)
        for i in range(len(sel_list)):
            if sel_list[i] == a: continue
            b = sel_list[i]
            for j in range(len(sel_list)):
                if sel_list[j] == a or sel_list[j] == b: continue
                c = sel_list[j]
                if (b + c) % a == target:
                    return False
    return True

def _greedy_build(order):
    """Greedy: add elements in given order if valid."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, time_limit=75.0):
    """Local search via additions and swaps to increase size."""
    best = list(selected)
    best_len = len(best)
    start = time.time()
    all_vals = set(range(1, 193))
    
    while time.time() - start < time_limit:
        improved = False
        cur = list(best)
        
        # Phase 1: Try to add a single element
        # Sort candidates descending, as larger numbers are often easier to add
        candidates = sorted(all_vals - set(cur), key=lambda x: -x)
        for v in candidates:
            if time.time() - start > time_limit:
                break
            if _can_add(cur, v):
                best = cur + [v]
                best_len += 1
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try remove 1, add 2 (net gain +1)
        # This is expensive, so we limit the search space
        found_2opt = False
        for i in range(len(cur)):
            if time.time() - start > time_limit:
                break
            removed = cur[i]
            base = [v for j, v in enumerate(cur) if j != i]
            
            # Try to add two elements to base
            # We check if base + [v1] is valid, then if base + [v1, v2] is valid
            # To save time, we only try top 15 candidates
            cands = sorted(all_vals - set(base), key=lambda x: -x)[:15]
            
            for v1 in cands:
                if time.time() - start > time_limit:
                    break
                if _can_add(base, v1):
                    base1 = base + [v1]
                    for v2 in cands:
                        if v2 == v1: continue
                        if time.time() - start > time_limit:
                            break
                        if _can_add(base1, v2):
                            new_S = base1 + [v2]
                            if len(new_S) > best_len:
                                best = new_S
                                best_len = len(best)
                                found_2opt = True
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
                
        if not improved:
            # No improvement found, break to avoid infinite loop
            break
            
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy with different orderings
    best = []
    
    # Descending (larger numbers first)
    greedy_desc = _greedy_build(range(192, 0, -1))
    if len(greedy_desc) > len(best):
        best = greedy_desc
        
    # Ascending
    greedy_asc = _greedy_build(range(1, 193))
    if len(greedy_asc) > len(best):
        best = greedy_asc
        
    # Random restarts
    rng = random.Random(42)
    for _ in range(50):
        if time.time() - start_time > _DEADLINE - 20:
            break
        order = list(range(1, 193))
        rng.shuffle(order)
        result = _greedy_build(order)
        if len(result) > len(best):
            best = result
            
    # Phase 2: Local search
    if time.time() - start_time < _DEADLINE - 5:
        time_left = _DEADLINE - 5 - (time.time() - start_time)
        best = _local_search(best, time_limit=time_left)
        
    return best
# EVOLVE-BLOCK-END
