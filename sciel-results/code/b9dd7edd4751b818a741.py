"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _check_dissociated(int_set, deadline):
    """Check if a set of integers has all subset sums distinct (dissociated).
    Uses the fact that we can multiply reciprocals by LCM to get integers.
    Returns True if all 2^n subset sums are distinct."""
    if time.time() > deadline:
        return False
    n = len(int_set)
    if n == 0:
        return True
    # Compute all 2^n subset sums using vectorized approach
    # For efficiency, use iterative doubling
    sums = np.array([0], dtype=np.int64)
    for i, val in enumerate(int_set):
        if time.time() > deadline:
            return False
        new_sums = sums + val
        sums = np.concatenate([sums, new_sums])
    # Check for duplicates
    if len(sums) != len(np.unique(sums)):
        return False
    return True

def _try_add_element(current_set, candidate, deadline):
    """Try adding candidate to current_set and check if still dissociated."""
    if time.time() > deadline:
        return False
    new_set = list(current_set) + [candidate]
    return _check_dissociated(new_set, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Seed set
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    all_elements = set(range(1, 23))
    
    # Phase 1: Try to add elements greedily
    improved = True
    while improved and time.time() < deadline:
        improved = False
        remaining = all_elements - set(best_set)
        for cand in sorted(remaining):
            if time.time() > deadline:
                break
            if _try_add_element(best_set, cand, deadline):
                best_set.append(cand)
                improved = True
                break  # restart the loop with updated set
    
    # Phase 2: Try swaps to potentially get larger sets
    # Try replacing one element with another to enable more additions
    swap_improved = True
    while swap_improved and time.time() < deadline:
        swap_improved = False
        remaining = all_elements - set(best_set)
        for i, elem in enumerate(best_set):
            if time.time() > deadline:
                break
            for cand in sorted(remaining):
                if time.time() > deadline:
                    break
                # Try swapping elem with cand
                new_set = best_set[:]
                new_set[i] = cand
                if _check_dissociated(new_set, deadline):
                    # Check if this swap enables adding more elements
                    # Try adding all remaining elements to the swapped set
                    for add_cand in sorted(remaining - {cand}):
                        if time.time() > deadline:
                            break
                        if _try_add_element(new_set, add_cand, deadline):
                            new_set.append(add_cand)
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        swap_improved = True
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
