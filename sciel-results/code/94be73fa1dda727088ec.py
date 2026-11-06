import time
import numpy as np
from itertools import combinations

def _max_sidon_size(arr):
    """Compute the largest weak Sidon subset size of arr using exact search.
    arr: sorted list of distinct integers.
    A weak Sidon set requires all unordered pair sums (including diagonals a+a) to be unique.
    """
    n = len(arr)
    if n <= 3:
        return n
    
    # Precompute all pair sums (i, j) with i <= j
    # For a subset to be weak Sidon, all pair sums must be distinct
    # We do a branch-and-bound / exact search for the largest valid subset
    
    # First, quickly check if the whole set is weak Sidon
    def is_weak_sidon(subset):
        m = len(subset)
        if m <= 3:
            return True
        sums = set()
        for i in range(m):
            for j in range(i, m):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Try to find largest subset via iterative deepening
    # Start from full set, remove elements to fix collisions
    # Use a greedy + backtracking approach
    
    # Fast check: if full set is Sidon, return n
    if is_weak_sidon(arr):
        return n
    
    # Try size n-1: remove each element, check if remainder is Sidon
    for i in range(n):
        subset = arr[:i] + arr[i+1:]
        if is_weak_sidon(subset):
            return n - 1
    
    # Try size n-2
    for i in range(n):
        for j in range(i+1, n):
            subset = arr[:i] + arr[i+1:j] + arr[j+1:]
            if is_weak_sidon(subset):
                return n - 2
    
    # Try size n-3
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                subset = arr[:i] + arr[i+1:j] + arr[j+1:k] + arr[k+1:]
                if is_weak_sidon(subset):
                    return n - 3
    
    # Shouldn't reach here for n=12 typically, but handle gracefully
    return n - 4

def _is_weak_sidon_fast(arr):
    """Fast check if arr is a weak Sidon set using numpy."""
    n = len(arr)
    if n <= 3:
        return True
    a = np.array(arr)
    # Compute all pair sums i <= j
    i_idx, j_idx = np.triu_indices(n)
    sums = a[i_idx] + a[j_idx]
    # Check for duplicates
    return len(np.unique(sums)) == len(sums)

def _max_sidon_size_fast(arr):
    """Compute largest weak Sidon subset size using fast checks."""
    n = len(arr)
    if n <= 3:
        return n
    
    # Check full set
    if _is_weak_sidon_fast(arr):
        return n
    
    # Check size n-1
    for i in range(n):
        subset = arr[:i] + arr[i+1:]
        if _is_weak_sidon_fast(subset):
            return n - 1
    
    # Check size n-2
    for i in range(n):
        for j in range(i+1, n):
            subset = arr[:i] + arr[i+1:j] + arr[j+1:]
            if _is_weak_sidon_fast(subset):
                return n - 2
    
    # Check size n-3
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                subset = arr[:i] + arr[i+1:j] + arr[j+1:k] + arr[k+1:]
                if _is_weak_sidon_fast(subset):
                    return n - 3
    
    return n - 4

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = None
    best_size = 12  # Worst case
    
    # Helper to evaluate a set
    def evaluate(s):
        return _max_sidon_size_fast(s)
    
    # Initial best: try the seed
    seed_set = list(range(12))
    seed_size = evaluate(seed_set)
    best_set = seed_set
    best_size = seed_size
    
    # Try some heuristic initial sets
    # Sets with high collision density
    heuristic_sets = [
        list(range(0, 128, 10))[:12],  # 0,10,20,...,110
        list(range(0, 128, 11))[:12],
        list(range(0, 128, 9))[:12],
        list(range(1, 128, 10))[:12],
        list(range(1, 128, 11))[:12],
        list(range(2, 128, 10))[:12],
        # Arithmetic progressions have many collisions
        list(range(0, 12, 1)),  # 0-11
        list(range(0, 12, 2)),  # 0,2,4,...,22
        list(range(0, 14, 2)),  # 0,2,4,...,12
        list(range(0, 16, 2)),  # 0,2,...,14
    ]
    
    for hs in heuristic_sets:
        if len(hs) == 12 and len(set(hs)) == 12 and all(0 <= x <= 128 for x in hs):
            sz = evaluate(hs)
            if sz < best_size:
                best_size = sz
                best_set = hs[:]
                if best_size <= 3:
                    return best_set
    
    # Local search with random restarts
    import random
    
    all_values = list(range(129))
    
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        # Random initial set
        current = random.sample(all_values, 12)
        current.sort()
        current_size = evaluate(current)
        
        if current_size < best_size:
            best_size = current_size
            best_set = current[:]
            if best_size <= 3:
                return best_set
        
        # Local search: try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_size = evaluate(current)
            
            for i in range(12):
                if time.time() > deadline:
                    break
                for new_val in all_values:
                    if new_val in current:
                        continue
                    # Try replacing current[i] with new_val
                    candidate = current[:]
                    candidate[i] = new_val
                    candidate.sort()
                    cand_size = evaluate(candidate)
                    if cand_size < current_size:
                        current = candidate
                        current_size = cand_size
                        improved = True
                        if current_size < best_size:
                            best_size = current_size
                            best_set = current[:]
                            if best_size <= 3:
                                return best_set
                        break  # Move to next i after a successful swap
    
    return best_set
