"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_can_add(x, selected):
    """Check if x can be added to selected without violating the condition.
    Only check pairs involving x since selected is already valid."""
    n = len(selected)
    if n < 2:
        return True
    
    # Check if x divides sum of any 3 elements in selected
    # (these 3 are distinct from x since x not in selected yet)
    for t in combinations(selected, 3):
        if (t[0] + t[1] + t[2]) % x == 0:
            return False
    
    # Check if any element a in selected divides sum of x and 2 other elements in selected
    for i, a in enumerate(selected):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + selected[j] + selected[k]
                if s % a == 0:
                    return False
    
    return True

def _check_validity(selected):
    """Check if no element divides the sum of any 3 distinct other selected elements."""
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if (t[0] + t[1] + t[2]) % a == 0:
                return False
    return True

def _greedy_construct(start_order, deadline):
    """Greedy construction: add elements in given order if valid."""
    selected = []
    for x in start_order:
        if time.time() > deadline:
            break
        if _check_can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Improve set by trying to add elements or swap elements to allow more additions."""
    best = list(selected)
    best_set = set(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any element
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            if _check_can_add(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break # Restart loop with new state
        
        if time.time() > deadline:
            break
            
        # Try swapping: remove one element, try to add elements
        if len(best) > 0 and random.random() < 0.3:
            # Pick a random element to remove
            idx = random.randint(0, len(best) - 1)
            temp_best = [e for i, e in enumerate(best) if i != idx]
            temp_set = set(temp_best)
            
            # Try to add elements to temp_best
            for x in range(192, 0, -1):
                if time.time() > deadline:
                    break
                if x in temp_set:
                    continue
                if _check_can_add(x, temp_best):
                    temp_best.append(x)
                    temp_set.add(x)
            
            if len(temp_best) > len(best):
                best = temp_best
                best_set = temp_set
                improved = True
                break

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial greedy from high to low
    order = list(range(192, 0, -1))
    best = _greedy_construct(order, deadline)
    
    # Randomized restarts
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        order = list(range(192, 0, -1))
        random.shuffle(order)
        candidate = _greedy_construct(order, deadline)
        if len(candidate) > len(best):
            best = candidate
    
    # Local search on the best found
    if time.time() < deadline:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Fallback if empty
    if len(best) == 0:
        best = [192]
        
    return best
# EVOLVE-BLOCK-END
