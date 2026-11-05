"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE_MARGIN = 5.0  # seconds before the 90s kill

def _can_add(x, selected):
    """Check if x can be added to selected while maintaining validity.
    
    Two types of conflicts:
    1. x divides sum of any 3 elements in selected.
    2. For some a in selected, there exist p,q in selected\{a} such that (x+p+q) % a == 0.
    """
    n = len(selected)
    if n < 3:
        # Type 1: no triples, so no conflict from x dividing a triple sum
        # Type 2: for each a, need pairs from selected\{a}; if n<3, at most 1 pair
        for a in selected:
            others = [v for v in selected if v != a]
            if len(others) >= 2:
                for i in range(len(others)):
                    for j in range(i+1, len(others)):
                        if (x + others[i] + others[j]) % a == 0:
                            return False
        return True
    
    # Type 1: check if x divides sum of any triple in selected
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if (selected[i] + selected[j] + selected[k]) % x == 0:
                    return False
    
    # Type 2: for each a in selected, check if there's a pair p,q in selected\{a} 
    # such that (x+p+q) % a == 0
    for idx_a, a in enumerate(selected):
        # Build list of others
        others = []
        for idx in range(n):
            if idx != idx_a:
                others.append(selected[idx])
        m = len(others)
        # Check all pairs
        for i in range(m):
            for j in range(i+1, m):
                if (x + others[i] + others[j]) % a == 0:
                    return False
    
    return True

def _greedy(order):
    """Greedy construction following the given order."""
    selected = []
    for x in order:
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _check_validity(selected):
    """Full validity check."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _local_improve(selected, all_candidates, deadline):
    """Try to improve by adding excluded elements."""
    best = list(selected)
    best_set = set(best)
    excluded = [x for x in all_candidates if x not in best_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_excluded = []
        for x in excluded:
            if time.time() > deadline:
                break
            if _can_add(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
            else:
                new_excluded.append(x)
        excluded = new_excluded
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    all_candidates = list(range(1, 193))
    
    best_solution = []
    best_len = 0
    
    # Descending order
    sol = _greedy(sorted(all_candidates, reverse=True))
    if len(sol) > best_len:
        best_len = len(sol)
        best_solution = sol
    
    # Ascending order  
    sol = _greedy(sorted(all_candidates))
    if len(sol) > best_len:
        best_len = len(sol)
        best_solution = sol
    
    # Random orderings
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        order = all_candidates[:]
        random.shuffle(order)
        sol = _greedy(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Local improvement on best solution
    if time.time() < deadline - 5:
        improved = _local_improve(best_solution, all_candidates, deadline - 3)
        if len(improved) > best_len:
            best_solution = improved
            best_len = len(improved)
    
    # More random restarts with local improvement
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        order = all_candidates[:]
        random.shuffle(order)
        sol = _greedy(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
            if time.time() < deadline - 3:
                improved = _local_improve(best_solution, all_candidates, deadline - 3)
                if len(improved) > best_len:
                    best_solution = improved
                    best_len = len(improved)
        elif len(sol) == best_len and time.time() < deadline - 10:
            # Try local improvement even if same length, might unlock more
            improved = _local_improve(sol, all_candidates, deadline - 3)
            if len(improved) > best_len:
                best_len = len(improved)
                best_solution = improved
    
    # Final validity check
    if not _check_validity(best_solution):
        # Fallback to a known valid solution
        best_solution = _greedy(sorted(all_candidates, reverse=True))
    
    return best_solution
# EVOLVE-BLOCK-END
