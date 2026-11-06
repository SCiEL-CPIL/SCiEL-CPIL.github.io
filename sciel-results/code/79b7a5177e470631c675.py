"""Optimized distinct subset sums with small largest element."""
# EVOLVE-BLOCK-START
import time

def _all_subset_sums_distinct(arr):
    """Check if all subset sums are distinct. Returns True if distinct.
    Uses early exit on first collision."""
    sums = {0}
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    # Greedy construction: a_1=1, a_{i+1} = smallest such that all subset sums remain distinct
    def greedy_construct(n):
        arr = [1]
        for i in range(1, n):
            candidate = arr[-1] + 1
            while True:
                if time.time() > deadline:
                    return None
                test_arr = arr + [candidate]
                if _all_subset_sums_distinct(test_arr):
                    arr.append(candidate)
                    break
                candidate += 1
        return arr
    
    # Try greedy
    best = greedy_construct(12)
    
    # If greedy didn't complete, fall back to powers of 2
    if best is None or len(best) < 12:
        best = [1 << i for i in range(12)]
    
    improved = list(best)
    
    # Local search: iteratively try to reduce elements
    while time.time() < deadline:
        # Sort indices by value descending to try reducing largest first
        idx_order = sorted(range(12), key=lambda i: -improved[i])
        found_improvement = False
        
        for idx in idx_order:
            if time.time() > deadline:
                break
            old_val = improved[idx]
            other_vals = set(improved[j] for j in range(12) if j != idx)
            
            # Try smaller values, skipping duplicates
            for new_val in range(old_val - 1, 0, -1):
                if new_val in other_vals:
                    continue
                improved[idx] = new_val
                if _all_subset_sums_distinct(improved):
                    found_improvement = True
                    break
                improved[idx] = old_val
            else:
                improved[idx] = old_val
                continue
            
            if found_improvement:
                break
        
        if not found_improvement:
            # Try 2-element modifications: reduce one, adjust another
            found_pair = False
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i+1, 12):
                    if time.time() > deadline:
                        break
                    vi, vj = improved[i], improved[j]
                    # Try reducing the larger element
                    if vj >= vi:
                        large_idx, small_idx = j, i
                    else:
                        large_idx, small_idx = i, j
                    large_val = improved[large_idx]
                    other_vals = set(improved[k] for k in range(12) if k != large_idx)
                    
                    for new_val in range(large_val - 1, 0, -1):
                        if new_val in other_vals:
                            continue
                        improved[large_idx] = new_val
                        if _all_subset_sums_distinct(improved):
                            found_pair = True
                            break
                        improved[large_idx] = large_val
                    if found_pair:
                        break
                if found_pair:
                    break
            
            if not found_pair:
                break  # Local minimum reached
    
    # Verify final result
    if not _all_subset_sums_distinct(improved):
        improved = [1 << i for i in range(12)]
    
    return improved
# EVOLVE-BLOCK-END
