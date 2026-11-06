"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _all_subset_sums_distinct(arr):
    """Check that all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _subset_sums_set(arr):
    """Return the set of all subset sums."""
    n = len(arr)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        sums.add(s)
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Start with powers of 2
    arr = [1 << i for i in range(12)]
    best_max = max(arr)
    best_arr = list(arr)
    
    # Verify initial solution is valid
    assert _all_subset_sums_distinct(best_arr)
    
    random.seed(42)
    
    # Local search: try to reduce individual elements
    # Key insight: if we have a superincreasing sequence, we can try to reduce elements
    # as long as the distinctness property holds
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort for processing (keep track of original indices)
        for i in range(11, -1, -1):  # Try reducing from largest to smallest
            if time.time() > deadline:
                break
            # Try reducing arr[i] by 1
            old_val = arr[i]
            if old_val <= 1:
                continue
            new_val = old_val - 1
            # Check distinctness and that new_val is positive and distinct from others
            if new_val <= 0:
                continue
            # Check new_val is not equal to any other element
            if any(arr[j] == new_val for j in range(12) if j != i):
                continue
            arr[i] = new_val
            if _all_subset_sums_distinct(arr):
                improved = True
                if max(arr) < best_max:
                    best_max = max(arr)
                    best_arr = list(arr)
            else:
                arr[i] = old_val
    
    # Additional refinement: try reducing by larger amounts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(11, -1, -1):
            if time.time() > deadline:
                break
            old_val = arr[i]
            if old_val <= 1:
                continue
            # Try reducing by various amounts
            for delta in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040]:
                if time.time() > deadline:
                    break
                new_val = old_val - delta
                if new_val <= 0:
                    break
                if any(arr[j] == new_val for j in range(12) if j != i):
                    continue
                arr[i] = new_val
                if _all_subset_sums_distinct(arr):
                    improved = True
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best_arr = list(arr)
                else:
                    arr[i] = old_val
                    break  # If reducing by delta fails, smaller deltas might work but let's try next element
    
    # Final check
    if _all_subset_sums_distinct(best_arr):
        return best_arr
    # Fallback to powers of 2
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
