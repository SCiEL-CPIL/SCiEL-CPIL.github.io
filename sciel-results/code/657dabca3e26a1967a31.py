"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _can_add(candidate, selected):
    """Check if candidate can be added to selected without violating constraints."""
    sel_set = set(selected)
    # Check 1: candidate doesn't divide sum of any 3 distinct others
    for t in combinations(selected, 3):
        if sum(t) % candidate == 0:
            return False
    # Check 2: no existing element divides sum of any 3 that include candidate + 2 others
    for a in selected:
        for t2 in combinations(selected, 2):
            s = candidate + t2[0] + t2[1]
            if s % a == 0:
                return False
    return True

def _is_valid_set(selected):
    """Full validation of a set."""
    sel_list = list(selected)
    for i, a in enumerate(sel_list):
        others = [x for j, x in enumerate(sel_list) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_with_order(order, deadline):
    """Greedy construction following a specific order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_improve(selected, deadline):
    """Try to improve the set by adding elements."""
    selected = list(selected)
    sel_set = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add elements not in set
        for x in range(1, 193):
            if x in sel_set:
                continue
            if _can_add(x, selected):
                selected.append(x)
                sel_set.add(x)
                improved = True
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    best = []
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy_with_order(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Also try descending order
    if time.time() < deadline:
        result = _greedy_with_order(list(range(192, 0, -1)), deadline)
        if len(result) > len(best):
            best = result
    
    # Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _is_valid_set(best):
        best = [192, 191, 190, 189, 188, 187, 186, 185, 184, 183]
    
    return best
# EVOLVE-BLOCK-END
