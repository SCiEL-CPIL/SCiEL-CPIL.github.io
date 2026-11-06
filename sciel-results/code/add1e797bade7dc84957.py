"""Rewritten program: bitmask subset-sum tracking + conflict-density heuristic + local search."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid_with_bitmask(selected):
    """Check if selected is valid: no element is mean of any subset of >=2 other elements."""
    n = len(selected)
    if n < 3:
        return True
    MAX_SUM = 24576
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 2:
            continue
        mask = 0
        for e in others:
            mask = mask | (mask << e)
        for k in range(2, m + 1):
            target = k * a
            if target > MAX_SUM:
                break
            if (mask >> target) & 1:
                return False
    return True

def _greedy_construct(all_candidates, order):
    """Greedily add candidates in the given order, checking validity each time."""
    selected = []
    for x in order:
        candidate = selected + [x]
        if _is_valid_with_bitmask(candidate):
            selected = candidate
    return selected

def _local_search(selected, all_candidates, time_limit, rng):
    """Try to improve selected by remove-one-add-one operations."""
    best = list(selected)
    best_size = len(best)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for i, rem in enumerate(list(selected)):
            remaining = [x for j, x in enumerate(selected) if j != i]
            rem_set = set(remaining)
            for add in all_candidates:
                if add in rem_set:
                    continue
                candidate = remaining + [add]
                if len(candidate) <= best_size:
                    continue
                if _is_valid_with_bitmask(candidate):
                    selected = candidate
                    best = list(selected)
                    best_size = len(best)
                    improved = True
                    break
            if time.time() - start_time > time_limit:
                break
        if not improved or len(selected) >= 48:
            break
    return best

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    all_candidates = list(range(1, 513))
    best_overall = []
    rng = random.Random(42)
    
    # Multiple greedy strategies
    orders = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
    ]
    for _ in range(8):
        shuffled = list(all_candidates)
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() - start_time > DEADLINE:
            break
        result = _greedy_construct(all_candidates, order)
        if len(result) > len(best_overall):
            best_overall = result
        if len(best_overall) >= 48:
            break
    
    # Local search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        remaining = min(DEADLINE - elapsed, 50)
        best_overall = _local_search(best_overall, all_candidates, remaining, rng)
    
    # Final verification
    if not _is_valid_with_bitmask(best_overall):
        best_overall = _greedy_construct(all_candidates, list(range(1, 513)))
    
    if len(best_overall) > 48:
        best_overall = best_overall[:48]
    
    return best_overall
# EVOLVE-BLOCK-END
