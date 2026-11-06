"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sunflower(a, b, c):
    """Check if three sets form a sunflower: all three pairwise intersections equal."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(existing, new_set):
    """Check if new_set can be added without creating a sunflower with any existing pair."""
    for i in range(len(existing)):
        a = existing[i]
        for j in range(i+1, len(existing)):
            b = existing[j]
            if _is_sunflower(a, b, new_set):
                return False
    return True

def _greedy_random(seed_val):
    """Greedy construction with random ordering."""
    rng = random.Random(seed_val)
    all_sets = list(combinations(range(1, 17), 4))
    rng.shuffle(all_sets)
    chosen = []
    for row in all_sets:
        s = set(row)
        if _can_add(chosen, s):
            chosen.append(s)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_family = None
    best_size = 0
    
    # Try multiple random seeds
    for seed in range(1000):
        if time.time() > deadline:
            break
        family = _greedy_random(seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            # If we hit the theoretical max, stop
            if best_size >= 250:
                break
    
    if best_family is None:
        # Fallback: deterministic greedy
        best_family = _greedy_random(0)
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
