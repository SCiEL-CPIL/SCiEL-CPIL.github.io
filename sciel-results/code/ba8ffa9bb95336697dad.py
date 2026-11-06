import time

def _subset_sums_distinct_fast(elements):
    """
    Check if all subset sums are distinct using a set-based approach.
    Returns True if all 2^n subset sums are unique.
    """
    sums = {0}
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns in sums:
                return False
            new_sums.add(ns)
        # Check for collisions within new_sums (shouldn't happen if sums were distinct)
        if len(new_sums) != len(sums):
            return False
        sums.update(new_sums)
    return True

def _compute_all_sums(elements):
    """Compute all subset sums as a set."""
    sums = {0}
    for e in elements:
        new_sums = {s + e for s in sums}
        sums.update(new_sums)
    return sums

def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 12
    
    # Phase 1: Greedy construction - try to build minimal sequence
    # For distinct subset sums, each new element a[i] must not be representable
    # as a sum of a subset of {a[0], ..., a[i-1]}
    # The minimum such value is 1 + max subset sum of previous = 1 + sum(prev)
    # which gives powers of 2. But we can search for better arrangements.
    
    # Start with the known good solution: powers of 2
    arr = [1 << i for i in range(n)]
    best_arr = list(arr)
    best_max = max(arr)
    
    # Verify
    if not _subset_sums_distinct_fast(best_arr):
        raise ValueError("Initial solution invalid")
    
    # Phase 2: Local search - try to reduce elements
    # Strategy: try reducing each element (from largest to smallest) 
    # by checking if the distinctness property still holds
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1, -1, -1):  # From largest index to smallest
            if time.time() > deadline:
                break
            old_val = arr[i]
            if old_val <= 1:
                continue
            
            # Try reducing by various amounts
            # First try small reductions
            for delta in range(1, min(old_val, 50)):
                if time.time() > deadline:
                    break
                new_val = old_val - delta
                if new_val <= 0:
                    break
                # Check distinctness with other elements
                if any(arr[j] == new_val for j in range(n) if j != i):
                    continue
                arr[i] = new_val
                if _subset_sums_distinct_fast(arr):
                    improved = True
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best_arr = list(arr)
                    break  # Found a reduction, try next element
                else:
                    arr[i] = old_val
                    break  # If delta fails, larger deltas won't help for this i
            
            # Try larger jumps
            if time.time() < deadline:
                for delta in [100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]:
                    if time.time() > deadline:
                        break
                    new_val = old_val - delta
                    if new_val <= 0:
                        continue
                    if any(arr[j] == new_val for j in range(n) if j != i):
                        continue
                    arr[i] = new_val
                    if _subset_sums_distinct_fast(arr):
                        improved = True
                        if max(arr) < best_max:
                            best_max = max(arr)
                            best_arr = list(arr)
                        break
                    else:
                        arr[i] = old_val
    
    # Phase 3: Try a different construction strategy
    # Greedy: at each step, pick the smallest positive integer not used
    # that maintains distinct subset sums
    if time.time() < deadline:
        greedy_arr = []
        used = set()
        for i in range(n):
            if time.time() > deadline:
                break
            candidate = 1
            while True:
                if time.time() > deadline:
                    break
                if candidate not in used:
                    test = greedy_arr + [candidate]
                    if _subset_sums_distinct_fast(test):
                        greedy_arr.append(candidate)
                        used.add(candidate)
                        break
                candidate += 1
            if time.time() > deadline:
                break
        if len(greedy_arr) == n and max(greedy_arr) < best_max:
            best_arr = greedy_arr
            best_max = max(greedy_arr)
    
    # Phase 4: Final polish - try to reduce the maximum element specifically
    if time.time() < deadline:
        # Sort to identify the maximum
        for _ in range(3):  # A few iterations
            if time.time() > deadline:
                break
            # Find index of max element
            max_idx = best_arr.index(max(best_arr))
            old_max = best_arr[max_idx]
            # Try reducing the max element
            for new_val in range(old_max - 1, 0, -1):
                if time.time() > deadline:
                    break
                if any(best_arr[j] == new_val for j in range(n) if j != max_idx):
                    continue
                test_arr = list(best_arr)
                test_arr[max_idx] = new_val
                if _subset_sums_distinct_fast(test_arr):
                    best_arr = test_arr
                    best_max = new_val
                    break
    
    # Final validation
    if _subset_sums_distinct_fast(best_arr):
        return best_arr
    else:
        # Fallback
        return [1 << i for i in range(n)]
