"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_sums_distinct(arr):
    """Check if all subset sums of arr are distinct."""
    n = len(arr)
    seen = set()
    # Use iterative subset sum generation
    sums = [0]
    for x in arr:
        new_sums = [s + x for s in sums]
        sums.extend(new_sums)
    if len(sums) != len(set(sums)):
        return False
    return True

def _validate(arr, n=12):
    """Check all conditions: n distinct positive ints, all subset sums distinct."""
    if len(arr) != n:
        return False
    if any(x <= 0 for x in arr):
        return False
    if len(set(arr)) != n:
        return False
    return _subset_sums_distinct(arr)

def _max_element(arr):
    return max(arr)

def _try_local_search(arr, max_time=55.0):
    """Local search to minimize max element."""
    start = time.time()
    best = list(arr)
    best_max = _max_element(best)
    
    n = len(arr)
    
    while time.time() - start < max_time:
        # Pick a random element to try to reduce
        idx = random.randint(0, n - 1)
        old_val = best[idx]
        
        # Try to reduce this element
        # It must stay positive, distinct from others, and maintain subset sum distinctness
        # Try values from old_val - 1 down to some lower bound
        other_vals = set(best) - {old_val}
        
        # The new value must be >= 1, not in other_vals
        # Also, for subset sum distinctness, a sufficient (not necessary) condition
        # is that it's less than or equal to sum of all others (necessary for non-trivial)
        # Actually, let's just try random smaller values
        
        tried = 0
        max_tries = 200
        while tried < max_tries:
            tried += 1
            # Try a random value smaller than old_val
            if old_val <= 1:
                break
            new_val = random.randint(1, old_val - 1)
            if new_val in other_vals:
                continue
            
            candidate = best[:]
            candidate[idx] = new_val
            
            if _subset_sums_distinct(candidate):
                if _max_element(candidate) < best_max:
                    best = candidate
                    best_max = _max_element(best)
                    break
                else:
                    # Accept even if not better, to explore
                    # But only if it doesn't increase max
                    if _max_element(candidate) <= best_max:
                        best = candidate
                        break
    
    return best, best_max

def _construct_superincreasing(n=12):
    """Generate superincreasing sequence: 1, 2, 4, 8, ..."""
    return [1 << i for i in range(n)]

def _random_valid_sequence(n=12):
    """Generate a random valid sequence using superincreasing then perturb."""
    seq = _construct_superincreasing(n)
    # Shuffle and try to reduce
    random.shuffle(seq)
    return seq

def construct():
    start = time.time()
    deadline = start + 58.0  # Safety margin before 60s
    
    # Start with superincreasing
    best = _construct_superincreasing(12)
    best_max = _max_element(best)  # 2048
    
    # Try multiple random restarts with local search
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create a perturbed starting point
        if restart == 0:
            arr = list(best)
        else:
            # Start from superincreasing with some random reductions
            arr = list(best)
            for i in range(12):
                if arr[i] > 1:
                    # Randomly reduce
                    reduction = random.randint(0, arr[i] // 3)
                    arr[i] = arr[i] - reduction
                    if arr[i] < 1:
                        arr[i] = 1
            
            # Check if still valid, if not, revert to best
            if not _subset_sums_distinct(arr):
                arr = list(best)
        
        # Local search
        arr, arr_max = _try_local_search(arr, max_time=min(1.0, deadline - time.time()))
        
        if arr_max < best_max:
            best = arr
            best_max = arr_max
    
    # Final check
    if not _validate(best, 12):
        # Fallback to superincreasing
        return _construct_superincreasing(12)
    
    return best
# EVOLVE-BLOCK-END
