"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def _subset_sums_distinct_fast(arr):
    """Faster check using incremental sums with set."""
    sums = {0}
    for x in arr:
        new_sums = {s + x for s in sums}
        if sums & new_sums:
            return False
        sums = sums | new_sums
    return True

def _check_valid_incremental(current_sums_set, new_elem):
    """Check if adding new_elem to current subset sums preserves distinctness."""
    # new_elem is invalid if it equals the difference of any two existing subset sums
    # i.e., if there exist s1, s2 in current_sums_set such that s1 - s2 = new_elem
    # This is equivalent to: (current_sums_set + new_elem) intersects current_sums_set
    # But faster: check if any (s + new_elem) is already in current_sums_set
    for s in current_sums_set:
        if s + new_elem in current_sums_set:
            return False
    return True

def _check_valid_incremental_v2(current_sums, new_elem):
    """Check using set operations."""
    current_set = set(current_sums)
    for s in current_sums:
        if s + new_elem in current_set:
            return False
    return True

def _validate(arr):
    """Validate: 12 distinct positive ints <= max, all subset sums distinct."""
    if len(arr) != 12:
        return False
    if len(set(arr)) != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if any(x > 1048576 for x in arr):
        return False
    return _subset_sums_distinct_fast(arr)

def _validate_fast(arr):
    """Fast validation assuming basic constraints already checked."""
    return _subset_sums_distinct_fast(arr)

def _score(arr):
    """Score is the largest element (lower is better)."""
    return max(arr)

def _try_reduce_max(arr, deadline):
    """Try to reduce elements in arr while maintaining validity."""
    arr = sorted(arr)
    n = len(arr)
    
    # Try to reduce each element from largest to smallest
    for target_idx in range(n - 1, -1, -1):
        current_val = arr[target_idx]
        # Try smaller values
        for new_val in range(current_val - 1, 0, -1):
            if time.time() > deadline:
                return arr
            if new_val in arr:
                continue
            test_arr = arr.copy()
            test_arr[target_idx] = new_val
            test_arr.sort()
            if _validate(test_arr):
                arr = test_arr
                break
        if time.time() > deadline:
            break
    
    return arr

def _greedy_construct(deadline=None):
    """Greedy construction: pick smallest valid element at each step."""
    if deadline is None:
        deadline = time.time() + 5
    
    n = 12
    elements = [1]
    subset_sums = [0, 1]
    subset_sums_set = set(subset_sums)
    
    for step in range(1, n):
        if time.time() > deadline:
            return None
        
        elem_set = set(elements)
        candidate = 1
        while True:
            if time.time() > deadline:
                return None
            if candidate not in elem_set:
                # Check if adding candidate preserves distinctness
                valid = True
                for s in subset_sums:
                    if s + candidate in subset_sums_set:
                        valid = False
                        break
                if valid:
                    elements.append(candidate)
                    subset_sums = subset_sums + [s + candidate for s in subset_sums]
                    subset_sums_set = set(subset_sums)
                    break
            candidate += 1
            if candidate > 1048576:
                return None
    
    return elements

def _local_search(arr, deadline):
    """Try to reduce elements while maintaining validity."""
    arr = sorted(arr)
    n = len(arr)
    best_arr = arr[:]
    best_max = max(arr)
    
    # Try to reduce each element from largest to smallest
    for target_idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
        current_val = arr[target_idx]
        # Try smaller values, skipping ones already in arr
        arr_set = set(arr)
        for new_val in range(current_val - 1, 0, -1):
            if time.time() > deadline:
                break
            if new_val in arr_set:
                continue
            test_arr = arr[:]
            test_arr[target_idx] = new_val
            test_arr.sort()
            if _validate(test_arr):
                arr = test_arr
                arr_set = set(arr)
                if max(arr) < best_max:
                    best_max = max(arr)
                    best_arr = arr[:]
                break
    
    return best_arr

def construct():
    deadline = time.time() + 85  # 85s out of 90s budget, 5s safety margin
    
    n = 12
    # Power of 2 solution as baseline
    best = [1 << i for i in range(n)]
    best_val = max(best)  # 2048
    
    # Try greedy construction
    if time.time() < deadline - 15:
        result = _greedy_construct(deadline - 15)
        if result is not None and _validate(result):
            val = max(result)
            if val < best_val:
                best_val = val
                best = result
    
    # Local improvement pass: try to reduce elements in the best solution
    if time.time() < deadline - 5:
        improved = _local_search(best, deadline - 3)
        if _validate(improved) and max(improved) < best_val:
            best = improved
            best_val = max(best)
    
    # Final validation
    if _validate(best):
        return best
    
    # Fallback: return power of 2 solution
    return [1 << i for i in range(n)]
