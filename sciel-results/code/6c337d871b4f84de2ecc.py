"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_valid(selected):
    """Check if the selected list is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for i, a in enumerate(selected):
        others = [x for x in selected if x != a]
        # Check all triples of others
        for j in range(len(others)):
            for k in range(j+1, len(others)):
                for l in range(k+1, len(others)):
                    s = others[j] + others[k] + others[l]
                    if s % a == 0:
                        return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    if x in selected:
        return False
    # Check x doesn't divide sum of any triple of existing selected
    n = len(selected)
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    s = selected[i] + selected[j] + selected[k]
                    if s % x == 0:
                        return False
    # Check existing elements don't divide sum of any triple including x
    for a in selected:
        # Triples that include x: pick 2 from selected (excluding a) and x
        others_excl_a = [v for v in selected if v != a]
        if len(others_excl_a) >= 2:
            for i in range(len(others_excl_a)):
                for j in range(i+1, len(others_excl_a)):
                    s = others_excl_a[i] + others_excl_a[j] + x
                    if s % a == 0:
                        return False
        # Triples of 3 from selected that don't include x are already valid
    return True

def _greedy_construct(order, deadline):
    """Greedy construction following given order."""
    selected = []
    sel_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in sel_set:
            continue
        if _can_add(selected, x):
            selected.append(x)
            sel_set.add(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve via adding, swapping, or removing elements."""
    selected = list(selected)
    sel_set = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding elements not in selected
        for x in range(192, 0, -1):
            if time.time() > deadline:
                return selected
            if x in sel_set:
                continue
            if _can_add(selected, x):
                selected.append(x)
                sel_set.add(x)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, try to add a better one
        for i, a in enumerate(selected):
            if time.time() > deadline:
                return selected
            remaining = selected[:i] + selected[i+1:]
            rem_set = sel_set - {a}
            for x in range(192, 0, -1):
                if x in rem_set:
                    continue
                if _can_add(remaining, x):
                    selected = remaining + [x]
                    sel_set = rem_set | {x}
                    improved = True
                    break
            if improved:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best = []
    
    # Strategy 1: Greedy from high to low
    order1 = list(range(192, 0, -1))
    result1 = _greedy_construct(order1, deadline)
    if len(result1) > len(best):
        best = result1
    
    if time.time() > deadline:
        return best
    
    # Local search on best so far
    best = _local_search(best, deadline)
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Greedy from low to high
    order2 = list(range(1, 193))
    result2 = _greedy_construct(order2, deadline)
    if len(result2) > len(best):
        best = result2
        best = _local_search(best, deadline)
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Random restarts with different orderings
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy_construct(order, deadline)
        if len(result) > len(best):
            best = result
            best = _local_search(best, deadline)
    
    if time.time() > deadline:
        return best
    
    # Final local search
    best = _local_search(best, deadline)
    
    return best
# EVOLVE-BLOCK-END
