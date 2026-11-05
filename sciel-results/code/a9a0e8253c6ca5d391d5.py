"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 256
DEADLINE = 55.0  # seconds, safety margin before 60s kill

def _is_sidon(selected):
    """Check if a list of integers forms a valid Sidon set."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _can_add(selected, sums, x):
    """Check if x can be added to selected while maintaining Sidon property."""
    new_sums = set()
    for a in selected:
        s = a + x
        if s in sums or s in new_sums:
            return False
        new_sums.add(s)
    s2 = 2 * x
    if s2 in sums or s2 in new_sums:
        return False
    new_sums.add(s2)
    return True

def _greedy_order(order):
    """Greedy construction following a given order of candidates.
    Runs to completion to guarantee the resulting set is maximal."""
    selected = []
    sums = set()
    for x in order:
        if _can_add(selected, sums, x):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
    return selected, sums

def _try_random_order(order):
    """Try a random permutation and return the result."""
    selected, sums = _greedy_order(order)
    return selected

def _is_maximal(selected):
    """Check if the set is maximal (no missing element can be added)."""
    if not selected:
        return False
    # Check Sidon property first
    if not _is_sidon(selected):
        return False
    
    selected_set = set(selected)
    sums = set()
    # Pre-compute sums for efficiency
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i, len(selected)):
            b = selected[j]
            sums.add(a + b)
            
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if x can be added
        can_add = True
        for a in selected:
            if (a + x) in sums:
                can_add = False
                break
        if can_add and (2 * x) in sums:
            can_add = False
            
        if can_add:
            return False
    return True

def construct():
    start_time = time.time()
    
    # First, try the natural ascending order (baseline)
    # _greedy_order now runs to completion, so it's fast
    best = _try_random_order(list(range(1, N + 1)))
    
    # Try descending order
    if time.time() < DEADLINE:
        cand = _try_random_order(list(range(N, 0, -1)))
        if len(cand) < len(best):
            best = cand
    
    # Try various random permutations
    rng = random.Random(42)
    attempts = 0
    while time.time() < DEADLINE - 2.0:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = _try_random_order(order)
        if len(cand) < len(best):
            best = cand
            # If we found a significantly smaller set, try to refine it
            # by attempting to remove elements
            if len(best) < 20:
                # Try to remove elements one by one
                improved = True
                while improved:
                    improved = False
                    for i in range(len(best)):
                        if time.time() > DEADLINE - 1.0:
                            break
                        trial = best[:i] + best[i+1:]
                        if _is_maximal(trial):
                            best = trial
                            improved = True
                            break
        attempts += 1
    
    # Final verification
    if not _is_maximal(best):
        # Fallback to a known valid construction (Ascending Greedy)
        best = _try_random_order(list(range(1, N + 1)))
    
    return sorted(best)

# EVOLVE-BLOCK-END
