"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_valid(selected):
    """Check if selected set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    if n < 3:
        return True
    # Check if x divides sum of any 3 existing elements
    for t in combinations(selected, 3):
        if sum(t) % x == 0:
            return False
    # Check if any existing element divides sum of x + any 2 other existing elements
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for pair in combinations(others, 2):
            s = x + pair[0] + pair[1]
            if s % a == 0:
                return False
    return True

def _greedy(order):
    """Greedy construction using given order of candidates."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve selected set via swaps and additions."""
    best = list(selected)
    best_size = len(best)
    all_nums = set(range(1, 193))
    
    while time.time() < deadline:
        improved = False
        
        # Try adding any missing number
        for x in sorted(all_nums - set(best), reverse=True):
            if _can_add(best, x):
                best.append(x)
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for i in range(len(best)):
            if time.time() >= deadline:
                break
            removed = best[i]
            for x in sorted(all_nums - set(best), reverse=True):
                if x == removed:
                    continue
                trial = best[:i] + best[i+1:] + [x]
                if _check_valid(trial):
                    best = trial
                    best_size = len(best)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Try multiple greedy orderings
    candidates = list(range(1, 193))
    orders = [
        list(range(192, 0, -1)),  # Descending
        list(range(1, 193)),      # Ascending
        list(range(192, 0, -2)),  # Even descending
        list(range(191, 0, -2)),  # Odd descending
    ]
    
    # Add some randomized orders
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        random.shuffle(candidates)
        orders.append(list(candidates))
    
    best_result = []
    for order in orders:
        if time.time() > deadline - 20:
            break
        result = _greedy(order)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search improvement
    if time.time() < deadline - 5:
        improved = _local_search(best_result, deadline)
        if len(improved) > len(best_result):
            best_result = improved
    
    return best_result
# EVOLVE-BLOCK-END
