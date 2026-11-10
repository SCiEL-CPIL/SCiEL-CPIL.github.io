"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_validity(selected):
    """Check if no element divides the sum of any 3 distinct other elements."""
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_construct():
    """Greedy construction from high to low."""
    selected = []
    for x in range(192, 0, -1):
        candidate = selected + [x]
        if _check_validity(candidate):
            selected.append(x)
    return selected

def _local_search(selected, deadline, max_rounds=50):
    """Local search with swaps to improve the set."""
    selected = list(selected)
    best = selected[:]
    best_size = len(selected)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Try swapping elements
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            for x in range(192, 0, -1):
                if x in selected:
                    continue
                candidate = selected[:]
                candidate[i] = x
                if _check_validity(candidate):
                    # Accept if it's valid
                    selected = candidate
                    improved = True
                    if len(selected) > best_size:
                        best = selected[:]
                        best_size = len(selected)
                    break
        
        # Try adding elements
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x not in selected:
                candidate = selected + [x]
                if _check_validity(candidate):
                    selected = candidate
                    improved = True
                    if len(selected) > best_size:
                        best = selected[:]
                        best_size = len(selected)
        
        if not improved:
            # Perturbation: remove a random element and try to fill
            if len(selected) > 1:
                idx = random.randint(0, len(selected) - 1)
                selected.pop(idx)
                for x in range(192, 0, -1):
                    if x not in selected:
                        candidate = selected + [x]
                        if _check_validity(candidate):
                            selected = candidate
                            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Initial greedy
    selected = _greedy_construct()
    best = selected[:]
    
    # Local search
    selected = _local_search(selected, deadline)
    if len(selected) > len(best):
        best = selected
    
    return best
# EVOLVE-BLOCK-END
