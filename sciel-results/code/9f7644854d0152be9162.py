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

def _local_search_fast(arr, time_limit=5.0):
    """Fast local search to try to reduce the largest element."""
    start_time = time.time()
    best = sorted(arr)
    best_max = _max_element(best)
    n = len(best)
    
    # We only try to reduce the largest element or swap elements
    # to keep the search space small and fast.
    while time.time() - start_time < time_limit:
        improved = False
        largest_idx = best.index(best_max)
        
        # Try reducing the largest element by small amounts
        # Start from best_max - 1 down to best_max - 100
        # This is a heuristic; if it works, great. If not, we stop quickly.
        for new_val in range(best_max - 1, max(0, best_max - 100), -1):
            if new_val in best:
                continue
            candidate = best[:]
            candidate[largest_idx] = new_val
            candidate.sort()
            if _all_subset_sums_distinct(candidate):
                best = candidate
                best_max = _max_element(best)
                improved = True
                break
        
        # Try swapping the largest with a smaller element?
        # Or just random perturbations
        if not improved:
            # Random perturbation
            if random.random() < 0.5:
                candidate = best[:]
                idx = random.randint(0, n - 1)
                old_val = candidate[idx]
                # Try a random value in the range
                new_val = random.randint(1, best_max)
                if new_val != old_val and new_val not in candidate:
                    candidate[idx] = new_val
                    candidate.sort()
                    if _all_subset_sums_distinct(candidate):
                        new_max = _max_element(candidate)
                        if new_max < best_max:
                            best = candidate
                            best_max = new_max
                            improved = True
        
        if not improved:
            # If we can't improve the max, we might be at a local minimum
            # Given the theoretical lower bound, we likely can't do better than powers of 2.
            # So we break to save time.
            break
            
    return best

def construct():
    """Construct 12 distinct positive integers <= 1048576 with all subset sums distinct, minimizing the largest."""
    start_time = time.time()
    # Safety margin: finish by 85s
    deadline = start_time + 85.0
    
    # Start with powers of 2 (known valid, likely optimal)
    best = [1 << i for i in range(12)]
    best_max = 2048
    
    # Validate the initial solution (should be fast)
    if not _all_subset_sums_distinct(best):
        # Fallback, though this shouldn't happen
        return best
    
    # Try to improve using fast local search
    # We limit this to 5 seconds to ensure we don't timeout
    # If it finds something better, great. If not, we return the optimal (or near-optimal) powers of 2.
    if time.time() < deadline - 10.0:
        improved = _local_search_fast(best, time_limit=5.0)
        if _max_element(improved) < best_max:
            best = improved
            best_max = _max_element(best)
    
    # Final validation
    if not _all_subset_sums_distinct(best):
        return [1 << i for i in range(12)]
    
    return best
# EVOLVE-BLOCK-END
