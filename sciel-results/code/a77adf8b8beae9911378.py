"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    candidate = selected + [x]
    n = len(candidate)
    # Check x against existing elements
    for i in range(n - 1):
        a = candidate[i]
        others = [candidate[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    # Check existing elements against x (already covered above when i < n-1)
    # But we need to check if any existing element divides sum of 3 others including x
    # The above loop checks all pairs where a is in candidate, so it's complete
    return True

def _greedy_descending():
    """Greedy from 192 down to 1."""
    selected = []
    for x in range(192, 0, -1):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _greedy_ascending():
    """Greedy from 1 up to 192."""
    selected = []
    for x in range(1, 193):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _greedy_random_order():
    """Greedy with random order."""
    candidates = list(range(1, 193))
    random.shuffle(candidates)
    selected = []
    for x in candidates:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, time_limit):
    """Local search: try to add elements, swap elements, or remove and re-add."""
    best = list(selected)
    best_size = len(best)
    start_time = time.time()
    
    # Precompute which elements are not in the set
    in_set = set(best)
    
    # Try adding elements
    while time.time() - start_time < time_limit * 0.4:
        improved = False
        candidates = [x for x in range(1, 193) if x not in in_set]
        random.shuffle(candidates)
        for x in candidates:
            if _can_add(best, x):
                best.append(x)
                in_set.add(x)
                best_size = len(best)
                improved = True
                break
        if not improved:
            break
    
    # Try swaps: remove one, add another
    while time.time() - start_time < time_limit * 0.7:
        improved = False
        for i in range(len(best)):
            for x in range(1, 193):
                if x in in_set:
                    continue
                # Try removing best[i] and adding x
                new_best = [best[j] for j in range(len(best)) if j != i]
                if _can_add(new_best, x):
                    new_best.append(x)
                    if _is_valid(new_best):
                        best = new_best
                        in_set = set(best)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Try multiple greedy strategies
    strategies = [
        _greedy_descending,
        _greedy_ascending,
        _greedy_random_order,
        _greedy_random_order,
        _greedy_random_order,
    ]
    
    best = []
    best_size = 0
    
    for strategy in strategies:
        if time.time() > deadline - 10.0:
            break
        result = strategy()
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Local search on the best result
    if time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        if remaining > 1.0:
            best = _local_search(best, remaining)
            best_size = len(best)
    
    return best
# EVOLVE-BLOCK-END
