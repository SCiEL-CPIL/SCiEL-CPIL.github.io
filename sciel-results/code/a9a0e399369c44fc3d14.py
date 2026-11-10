import time
import random
from itertools import combinations

def _all_subset_sums_unique(arr):
    """Check that all 2^n subset sums are distinct."""
    n = len(arr)
    total = len(arr)
    sums = set()
    # Use bitmask enumeration
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _local_search(arr, max_time, rng):
    """Local search to minimize the largest element while maintaining distinct subset sums."""
    n = len(arr)
    best = arr[:]
    best_max = max(arr)
    start = time.time()
    
    while time.time() - start < max_time:
        # Try to reduce each element
        improved = False
        for i in range(n):
            orig = arr[i]
            # Try to reduce arr[i]
            for delta in range(1, min(orig, 100) + 1):
                new_val = orig - delta
                if new_val < 1:
                    break
                arr[i] = new_val
                # Check distinctness and uniqueness
                if len(set(arr)) < n:
                    arr[i] = orig
                    continue
                if _all_subset_sums_unique(arr):
                    improved = True
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best = arr[:]
                    break
            if arr[i] != orig:
                pass  # already updated
            else:
                arr[i] = orig  # restore
        
        # Try swapping or small perturbations
        for i in range(n):
            for delta in range(1, 20):
                new_val = arr[i] + delta
                if new_val > 1048576:
                    break
                old_val = arr[i]
                arr[i] = new_val
                if len(set(arr)) < n:
                    arr[i] = old_val
                    continue
                if _all_subset_sums_unique(arr):
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best = arr[:]
                else:
                    arr[i] = old_val
        
        if not improved:
            # Try random restarts with small perturbations
            if time.time() - start > max_time * 0.8:
                break
    
    return best

def _greedy_construct(n):
    """Greedy construction: pick smallest possible value at each step that maintains distinct subset sums."""
    arr = []
    for i in range(n):
        # Try values starting from 1
        for val in range(1, 1048577):
            candidate = arr + [val]
            if len(set(candidate)) < len(candidate):
                continue
            if _all_subset_sums_unique(candidate):
                arr.append(val)
                break
    return arr

def _simulated_annealing(arr, max_time, rng):
    """Simulated annealing to minimize max element."""
    n = len(arr)
    current = arr[:]
    current_max = max(arr)
    best = arr[:]
    best_max = current_max
    start = time.time()
    T = 10.0
    cooling = 0.995
    
    while time.time() - start < max_time:
        i = rng.randint(0, n - 1)
        delta = rng.randint(-50, 50)
        if delta == 0:
            continue
        new_val = current[i] + delta
        if new_val < 1 or new_val > 1048576:
            continue
        
        old_val = current[i]
        current[i] = new_val
        
        if len(set(current)) < n or not _all_subset_sums_unique(current):
            # Revert
            current[i] = old_val
            T *= cooling
            continue
        
        new_max = max(current)
        delta_energy = new_max - current_max
        
        if delta_energy <= 0 or rng.random() < (1.0 / (1.0 + (1.0 - rng.random()) * T)) and delta_energy < T:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best = current[:]
        else:
            current[i] = old_val
        
        T *= cooling
    
    return best

def construct():
    start = time.time()
    deadline = 85.0  # Leave 5s safety margin
    n = 12
    rng = random.Random(42)
    
    # Phase 1: Greedy construction (fast, gives a valid solution)
    arr = _greedy_construct(n)
    if len(arr) < n:
        # Fallback to powers of 2
        arr = [1 << i for i in range(n)]
    
    best = arr[:]
    best_max = max(arr)
    
    elapsed = time.time() - start
    if elapsed > deadline - 30:
        return best
    
    # Phase 2: Local search to improve
    improved = _local_search(arr, min(30.0, deadline - (time.time() - start) - 5), rng)
    if max(improved) < best_max:
        best = improved
        best_max = max(improved)
    
    elapsed = time.time() - start
    if elapsed > deadline - 20:
        return best
    
    # Phase 3: Simulated annealing for further refinement
    sa_result = _simulated_annealing(best, min(20.0, deadline - (time.time() - start) - 3), rng)
    if max(sa_result) < best_max:
        best = sa_result
        best_max = max(sa_result)
    
    # Final verification
    assert len(best) == n
    assert all(isinstance(x, int) and x > 0 for x in best)
    assert max(best) <= 1048576
    assert len(set(best)) == n
    assert _all_subset_sums_unique(best)
    
    return best
