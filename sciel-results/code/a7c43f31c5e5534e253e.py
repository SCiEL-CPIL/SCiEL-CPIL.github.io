"""Optimized minimal maximal no-isosceles set constructor."""
import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_MARGIN = 5.0  # seconds before the 330s limit

def _dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def _is_isosceles(p, a, b):
    d1 = _dist2(p, a)
    d2 = _dist2(p, b)
    d3 = _dist2(a, b)
    return d1 == d2 or d1 == d3 or d2 == d3

def _greedy_maximal(order):
    """Build a maximal no-isosceles set by greedily adding points in given order."""
    chosen = []
    for p in order:
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _is_isosceles(p, chosen[i], chosen[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(p)
    return chosen

def _try_remove(chosen):
    """Try to remove points one at a time while maintaining maximality.
    Returns (new_set, removed_count)."""
    current = list(chosen)
    removed = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            # Try removing current[i]
            candidate = current[:i] + current[i+1:]
            # Check if candidate is still maximal: no point can be added
            is_maximal = True
            for p in itertools.product(range(N), repeat=2):
                if p in candidate:
                    continue
                # Check if p can be added (no isosceles with any pair in candidate)
                can_add = True
                for a in range(len(candidate)):
                    for b in range(a + 1, len(candidate)):
                        if _is_isosceles(p, candidate[a], candidate[b]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    is_maximal = False
                    break
            if is_maximal:
                current = candidate
                removed += 1
                changed = True
                break  # restart scanning from beginning
    return current, removed

def _local_search(chosen, time_limit):
    """Local search with swap moves to minimize set size while maintaining maximality."""
    current = list(chosen)
    best = list(current)
    best_size = len(best)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Try removing each point
        improved = False
        for i in range(len(current)):
            candidate = current[:i] + current[i+1:]
            # Check maximality
            is_maximal = True
            for p in itertools.product(range(N), repeat=2):
                if p in candidate:
                    continue
                can_add = True
                for a in range(len(candidate)):
                    for b in range(a + 1, len(candidate)):
                        if _is_isosceles(p, candidate[a], candidate[b]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    is_maximal = False
                    break
            if is_maximal:
                current = candidate
                if len(current) < best_size:
                    best = list(current)
                    best_size = len(best)
                improved = True
                break
        if not improved:
            break
    return best

def _check_maximal(chosen):
    """Check if chosen is a maximal no-isosceles set."""
    for p in itertools.product(range(N), repeat=2):
        if p in chosen:
            continue
        can_add = True
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                if _is_isosceles(p, chosen[a], chosen[b]):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 330.0 - DEADLINE_MARGIN
    
    # Generate multiple random orderings and find the best greedy result
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_set = None
    best_size = float('inf')
    
    # Try several random orderings
    rng = random.Random(42)
    max_greedy_trials = 50
    
    for trial in range(max_greedy_trials):
        if time.time() > deadline - 60:
            break
        
        order = all_points[:]
        rng.shuffle(order)
        result = _greedy_maximal(order)
        
        if len(result) < best_size:
            best_size = len(result)
            best_set = result
    
    # Also try the standard row-major order
    result = _greedy_maximal(all_points)
    if len(result) < best_size:
        best_size = len(result)
        best_set = result
    
    # Apply local search to the best set
    if best_set is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            best_set = _local_search(best_set, min(remaining_time - 5, 120))
    
    # Final safety: if we haven't found anything, use the row-major greedy
    if best_set is None:
        best_set = _greedy_maximal(all_points)
    
    return best_set
