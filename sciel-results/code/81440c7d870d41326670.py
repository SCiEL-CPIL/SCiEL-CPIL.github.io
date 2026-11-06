"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _all_subset_sums_distinct(arr):
    """Check that all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    # Use bit manipulation to enumerate all subsets
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _max_element(arr):
    return max(arr)

def _greedy_construct(n=12):
    """Greedy: pick smallest possible value at each step that keeps subset sums distinct."""
    arr = []
    for i in range(n):
        # Try values starting from 1
        for val in range(1, 1048577):
            candidate = arr + [val]
            if _all_subset_sums_distinct(candidate):
                arr = candidate
                break
        else:
            break
    return arr

def _local_search(arr, time_limit=60.0, max_iters=50000):
    """Local search: try to reduce the largest element by swapping/replacing elements."""
    start_time = time.time()
    best = list(arr)
    best_max = _max_element(best)
    
    n = len(best)
    
    # Sort for consistency
    best.sort()
    
    iters = 0
    while iters < max_iters:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
            break
        iters += 1
        
        # Try to reduce the largest element
        largest_idx = best.index(best_max)
        
        # Try replacing the largest element with a smaller value
        found_improvement = False
        # Try values from best_max - 1 down to 1
        for new_val in range(best_max - 1, 0, -1):
            if new_val in best:
                continue
            candidate = best[:]
            candidate[largest_idx] = new_val
            candidate.sort()
            if _all_subset_sums_distinct(candidate):
                # Check if this reduces the max
                new_max = _max_element(candidate)
                if new_max < best_max:
                    best = candidate
                    best_max = new_max
                    found_improvement = True
                    break
        
        if not found_improvement:
            # Try more complex moves: swap two elements
            for i in range(n):
                for j in range(i + 1, n):
                    # Try swapping
                    candidate = best[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if _all_subset_sums_distinct(candidate):
                        # Check if max decreased
                        new_max = _max_element(candidate)
                        if new_max < best_max:
                            best = candidate
                            best.sort()
                            best_max = new_max
                            found_improvement = True
                            break
                if found_improvement:
                    break
        
        if not found_improvement:
            # Try reducing any element (not just the largest) to enable further reductions
            improved_any = False
            for i in range(n):
                for new_val in range(best[i] - 1, 0, -1):
                    if new_val in best:
                        continue
                    candidate = best[:]
                    candidate[i] = new_val
                    candidate.sort()
                    if _all_subset_sums_distinct(candidate):
                        new_max = _max_element(candidate)
                        if new_max <= best_max:
                            best = candidate
                            best_max = new_max
                            improved_any = True
                            break
                if improved_any:
                    break
            
            if not improved_any:
                # Stuck in local minimum, try random perturbation
                if random.random() < 0.3:
                    # Random restart with small perturbation
                    candidate = best[:]
                    idx = random.randint(0, n - 1)
                    old_val = candidate[idx]
                    # Try a random different value
                    for _ in range(100):
                        new_val = random.randint(1, best_max)
                        if new_val == old_val or new_val in candidate:
                            continue
                        candidate[idx] = new_val
                        candidate.sort()
                        if _all_subset_sums_distinct(candidate):
                            best = candidate
                            best_max = _max_element(best)
                            break
    
    return best

def construct():
    """Construct 12 distinct positive integers <= 1048576 with all subset sums distinct, minimizing the largest."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Start with powers of 2 (known valid)
    best = [1 << i for i in range(12)]
    best_max = 2048
    
    # Try greedy construction
    if time.time() < deadline - 10.0:
        greedy = _greedy_construct(12)
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
