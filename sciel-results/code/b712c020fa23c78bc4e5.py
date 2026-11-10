"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _validate(arr):
    """Check that arr is a valid superincreasing-like set with distinct subset sums."""
    n = len(arr)
    if n != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if len(set(arr)) != n:
        return False
    # Check all subset sums are distinct
    # Use a set to track sums
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

def _validate_fast(arr):
    """Faster validation using incremental subset sum enumeration."""
    n = len(arr)
    if n != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if len(set(arr)) != n:
        return False
    
    # Generate all subset sums incrementally
    sums = {0}
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _try_local_search(arr, max_iter=50000, deadline=None):
    """Local search: try to reduce the maximum element by swapping/replacing elements."""
    best = list(arr)
    best_max = max(best)
    n = len(best)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to reduce the largest element
        # Find the index of the max element
        max_idx = best.index(max(best))
        
        # Try replacing the max element with a smaller value
        # The new value must maintain distinct subset sums
        # Strategy: try values slightly below the current max
        current_max = best[max_idx]
        
        # Try a range of candidate values below current_max
        # To be efficient, try values that might work
        found_improvement = False
        
        # Try random candidates below current_max
        for _ in range(100):
            if deadline is not None and time.time() > deadline:
                break
            
            # Pick a random candidate
            low = max(1, current_max - max(1, current_max // 10))
            high = current_max - 1
            if low >= high:
                continue
            
            candidate = random.randint(low, high)
            if candidate in best:
                continue
            
            new_arr = list(best)
            new_arr[max_idx] = candidate
            new_arr.sort()
            
            if _validate_fast(new_arr):
                best = new_arr
                best_max = max(best)
                found_improvement = True
                break
        
        if not found_improvement:
            # Try other local moves: swap two elements
            if random.random() < 0.3:
                i, j = random.sample(range(n), 2)
                new_arr = list(best)
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
                if _validate_fast(new_arr) and max(new_arr) < best_max:
                    best = new_arr
                    best_max = max(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Start with a known valid solution: powers of 2
    # But we can do better. Let's start with a greedy construction.
    
    # Greedy: start with 1, then each next element is sum of all previous + 1
    # This gives the minimal possible superincreasing sequence
    # For n=12: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Max = 2048
    
    # But we can do better with non-superincreasing sequences.
    # Let's try a randomized approach.
    
    best_solution = [1 << i for i in range(12)]
    best_max = max(best_solution)
    
    # Try to improve via local search
    result = _try_local_search(best_solution, max_iter=200000, deadline=deadline)
    if max(result) < best_max:
        best_solution = result
        best_max = max(best_solution)
    
    # Try more iterations if time permits
    if time.time() < deadline - 5:
        result = _try_local_search(best_solution, max_iter=200000, deadline=deadline)
        if max(result) < best_max:
            best_solution = result
            best_max = max(best_solution)
    
    # Additional refinement: try to reduce individual large elements
    if time.time() < deadline - 3:
        n = len(best_solution)
        for _ in range(50):
            if time.time() > deadline:
                break
            max_idx = best_solution.index(max(best_solution))
            current_max = best_solution[max_idx]
            improved = False
            # Try decreasing the max element by small amounts
            for delta in [1, 2, 3, 5, 8, 13, 21]:
                candidate = current_max - delta
                if candidate <= 0 or candidate in best_solution:
                    continue
                new_arr = list(best_solution)
                new_arr[max_idx] = candidate
                new_arr.sort()
                if _validate_fast(new_arr):
                    best_solution = new_arr
                    best_max = max(best_solution)
                    improved = True
                    break
            if not improved:
                break
    
    return best_solution

# EVOLVE-BLOCK-END
