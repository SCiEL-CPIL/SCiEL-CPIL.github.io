"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE_SECONDS = 85  # safety margin before 90s kill

def _can_add(selected, x):
    """Check if x can be added to selected without violating the constraint."""
    candidate = selected + [x]
    n = len(candidate)
    # For each element a in candidate, check no 3 distinct OTHER elements sum to a multiple of a
    for i, a in enumerate(candidate):
        others = [candidate[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        # Check all triples from others
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _is_valid(selected):
    """Verify the full set is valid."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy(order, deadline):
    """Greedy construction following the given order."""
    selected = []
    for x in order:
        if time.time() >= deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best = []
    
    # Strategy 1: Descending order (original)
    order1 = list(range(192, 0, -1))
    result = _greedy(order1, deadline)
    if len(result) > len(best):
        best = result
    
    if time.time() >= deadline:
        return best
    
    # Strategy 2: Randomized restarts
    random.seed(42)
    for attempt in range(200):
        if time.time() >= deadline - 2:
            break
        
        # Create a random perturbation of the descending order
        base = list(range(192, 0, -1))
        # Randomly shuffle a portion to create diversity
        shuffle_start = random.randint(0, 100)
        shuffle_end = random.randint(shuffle_start + 1, 192)
        segment = base[shuffle_start:shuffle_end]
        random.shuffle(segment)
        base[shuffle_start:shuffle_end] = segment
        
        result = _greedy(base, deadline)
        if len(result) > len(best):
            best = result
    
    # Final validation
    if best and not _is_valid(best):
        # Fallback to deterministic descending
        best = _greedy(list(range(192, 0, -1)), deadline)
    
    return best
# EVOLVE-BLOCK-END
