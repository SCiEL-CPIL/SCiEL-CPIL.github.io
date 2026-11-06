"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(a, existing_sorted):
    """Check if adding a to existing_sorted (sorted list) maintains validity.
    For all b <= c in existing_sorted with b > a: a must not divide (b+c).
    """
    # Find first index where element > a
    lo = 0
    hi = len(existing_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if existing_sorted[mid] > a:
            hi = mid
        else:
            lo = mid + 1
    start = lo
    if start >= len(existing_sorted):
        return True
    
    sub = existing_sorted[start:]
    n_sub = len(sub)
    if n_sub < 2:
        return True
    
    # For each i from 0 to n_sub-1, b = sub[i]
    # For each j from i to n_sub-1, c = sub[j]
    # Check (b + c) % a != 0
    # Equivalent: c % a != (-b) % a
    for i in range(n_sub):
        b = sub[i]
        target = (-b) % a
        # Check if any sub[j] for j >= i has sub[j] % a == target
        for j in range(i, n_sub):
            if sub[j] % a == target:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 256
    best_set = list(range(192, n+1))
    best_size = len(best_set)
    
    # Note: 1 cannot be in any valid set of size >= 2
    # Greedy from large to small, skipping 1
    current = []
    for a in range(n, 1, -1):
        if _can_add(a, current):
            current.append(a)
        if time.time() > deadline:
            break
    if len(current) > best_size:
        best_set = current
        best_size = len(current)
    
    # Try different strategies: start with a subset of large numbers, add smaller
    for start_thresh in range(160, 257, 8):
        if time.time() > deadline:
            break
        current = list(range(start_thresh, n+1))
        for a in range(start_thresh - 1, 1, -1):
            if _can_add(a, current):
                current.append(a)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Local search: try to improve by swaps
    if time.time() < deadline - 15:
        current = list(best_set)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                removed = current[i]
                trial = [x for x in current if x != removed]
                # Try to add a number not in trial that gives larger set
                for candidate in range(n, 1, -1):
                    if candidate in trial:
                        continue
                    if _can_add(candidate, trial):
                        trial.append(candidate)
                        if len(trial) > len(current):
                            current = trial
                            improved = True
                            break
    
    # Final pass: try to add any remaining numbers
    if time.time() < deadline - 3:
        current = list(best_set)
        for a in range(n, 1, -1):
            if a in current:
                continue
            if _can_add(a, current):
                current.append(a)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    return best_set
# EVOLVE-BLOCK-END
