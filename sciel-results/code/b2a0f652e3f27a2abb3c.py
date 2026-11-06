"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _all_subset_sums_distinct(arr):
    """Check that all 2^n subset sums are distinct using vectorized computation."""
    n = len(arr)
    if n == 0:
        return True
    arr = np.array(arr, dtype=np.int64)
    sums = np.zeros(1 << n, dtype=np.int64)
    for i in range(n):
        half = 1 << i
        block = 1 << (i + 1)
        for start in range(0, 1 << n, block):
            sums[start + half:start + block] += arr[i]
    sorted_sums = np.sort(sums)
    return np.all(np.diff(sorted_sums) > 0)

def _max_element(arr):
    return max(arr)

def _greedy_construct(n=12, deadline=None):
    """Greedy: pick smallest possible value at each step that keeps subset sums distinct."""
    arr = []
    for i in range(n):
        if deadline is not None and time.time() > deadline:
            break
        # The minimum valid next value is sum(arr) + 1 (superincreasing bound)
        # But we can try smaller values if they work
        min_val = sum(arr) + 1 if arr else 1
        # Try from 1 upward, but start from min_val for efficiency in later steps
        start = 1 if i < 3 else min_val
        for val in range(start, 1048577):
            candidate = arr + [val]
            if _all_subset_sums_distinct(candidate):
                arr = candidate
                break
        else:
            break
    return arr

def _local_search(arr, time_limit=60.0, max_iters=50000):
    """Local search: try to reduce the largest element by various moves."""
    start_time = time.time()
    best = sorted(arr)
    best_max = _max_element(best)
    n = len(best)
    
    iters = 0
    while iters < max_iters:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
            break
        iters += 1
        
        found_improvement = False
        
        # Move 1: Try reducing the largest element
        largest_idx = best.index(best_max)
        # Sample values to try, focusing on the range just below current max
        candidates_to_try = set()
        # Try values near the current max
        for offset in range(1, 50):
            candidates_to_try.add(best_max - offset)
        # Try some random values in a wider range
        for _ in range(20):
            candidates_to_try.add(random.randint(1, best_max - 1))
        
        for new_val in sorted(candidates_to_try, reverse=True):
            if new_val <= 0 or new_val in best:
                continue
            candidate = best[:]
            candidate[largest_idx] = new_val
            candidate.sort()
            if _all_subset_sums_distinct(candidate):
                new_max = _max_element(candidate)
                if new_max < best_max:
                    best = candidate
                    best_max = new_max
                    found_improvement = True
                    break
        
        # Move 2: Try reducing any element
        if not found_improvement:
            for i in range(n):
                for new_val in range(best[i] - 1, max(0, best[i] - 30), -1):
                    if new_val <= 0 or new_val in best:
                        continue
                    candidate = best[:]
                    candidate[i] = new_val
                    candidate.sort()
                    if _all_subset_sums_distinct(candidate):
                        new_max = _max_element(candidate)
                        if new_max <= best_max:
                            best = candidate
                            best_max = new_max
                            found_improvement = True
                            break
                if found_improvement:
                    break
        
        # Move 3: Random perturbation to escape local minima
        if not found_improvement:
            if random.random() < 0.3:
                candidate = best[:]
                num_changes = random.randint(1, 2)
                valid = True
                for _ in range(num_changes):
                    idx = random.randint(0, n - 1)
                    old_val = candidate[idx]
                    found = False
                    for _ in range(100):
                        new_val = random.randint(1, max(best_max, 50))
                        if new_val == old_val or new_val in candidate:
                            continue
                        candidate[idx] = new_val
                        found = True
                        break
                    if not found:
                        valid = False
                        break
                if valid:
                    candidate.sort()
                    if _all_subset_sums_distinct(candidate):
                        new_max = _max_element(candidate)
                        if new_max <= best_max:
                            best = candidate
                            best_max = new_max
                            found_improvement = True
    
    return best

def construct():
    """Construct 12 distinct positive integers <= 1048576 with all subset sums distinct, minimizing the largest."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Start with powers of 2 (known valid)
    best = [1 << i for i in range(12)]
    best_max = 2048
    
    # Try greedy construction
    if time.time() < deadline - 20.0:
        greedy = _greedy_construct(12, deadline=deadline - 20.0)
        if len(greedy) == 12:
            g_max = _max_element(greedy)
            if g_max < best_max:
                best = greedy
                best_max = g_max
    
    # Local search to improve
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        improved = _local_search(best, time_limit=remaining)
        if _max_element(improved) < best_max:
            best = improved
            best_max = _max_element(best)
    
    # Final validation
    if not _all_subset_sums_distinct(best):
        # Fallback to powers of 2
        return [1 << i for i in range(12)]
    
    return best
# EVOLVE-BLOCK-END
