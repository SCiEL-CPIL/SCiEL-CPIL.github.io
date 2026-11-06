"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _greedy_find(order, vals):
    """Greedy: add elements in order if they don't cause collision."""
    current_sums = {0}
    selected = []
    for a in order:
        v = vals[a]
        new_sums = set()
        ok = True
        for s in current_sums:
            ns = s + v
            if ns in current_sums or ns in new_sums:
                ok = False
                break
            new_sums.add(ns)
        if ok:
            current_sums |= new_sums
            selected.append(a)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    vals = {a: _L // a for a in range(1, 23)}
    
    # Phase 1: Greedy to get a good starting point
    best_set = []
    for order in [list(range(1, 23)), list(range(22, 0, -1))]:
        gs = _greedy_find(order, vals)
        if len(gs) > len(best_set):
            best_set = gs
    
    best_size = len(best_set)
    
    # Phase 2: Backtracking to try to improve
    def check_and_add(current_sums, v):
        new_sums = set()
        for s in current_sums:
            ns = s + v
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        return new_sums
    
    def backtrack(order, idx, current_sums, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        if idx >= len(order):
            return
        # Pruning: even if we include all remaining, can't beat best
        if len(current_set) + (len(order) - idx) <= best_size:
            return
        a = order[idx]
        v = vals[a]
        new_sums = check_and_add(current_sums, v)
        if new_sums is not None:
            backtrack(order, idx + 1, current_sums | new_sums, current_set + [a])
        backtrack(order, idx + 1, current_sums, current_set)
    
    # Try to find a larger set
    for order in [list(range(1, 23)), list(range(22, 0, -1))]:
        if time.time() > deadline:
            break
        backtrack(order, 0, {0}, [])
    
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return sorted(best_set)
# EVOLVE-BLOCK-END
