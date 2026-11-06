"""B2[2] set construction: optimized greedy + incremental local search with numpy counts."""
import time
import random
import numpy as np

def _greedy_build():
    """
    Greedy construction using a numpy array for sum counts.
    Add x if no sum exceeds 2 representations.
    """
    counts = np.zeros(2001, dtype=np.int32)  # sums range from 2 to 2000
    chosen = []
    for x in range(1, 1001):
        # Check if adding x would violate B2[2]
        # New sums: x + y for y in chosen, and 2*x
        # We need to check if any of these sums already have count >= 2
        violation = False
        for y in chosen:
            if counts[x + y] >= 2:
                violation = True
                break
        if not violation and counts[2 * x] >= 2:
            violation = True
        
        if not violation:
            # Update counts
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
    
    return chosen, counts

def _build_counts(set_list):
    """Return numpy array: sum -> count of pairs (a,b) with a<=b from set_list."""
    counts = np.zeros(2001, dtype=np.int32)
    n = len(set_list)
    for i in range(n):
        for j in range(i, n):
            sm = set_list[i] + set_list[j]
            counts[sm] += 1
    return counts

def _try_add(set_list, x, counts):
    """
    Try adding x to set_list (sorted). Uses numpy counts.
    Returns (success, new_set_list, new_counts).
    """
    if x in set_list:
        return False, set_list, counts
    
    # Check validity
    for y in set_list:
        if counts[x + y] >= 2:
            return False, set_list, counts
    if counts[2 * x] >= 2:
        return False, set_list, counts
    
    # Valid: create new set and counts
    new_set = list(set_list)
    new_set.append(x)
    new_set.sort()
    
    new_counts = counts.copy()
    for y in set_list:
        new_counts[x + y] += 1
    new_counts[2 * x] += 1
    
    return True, new_set, new_counts

def _remove(set_list, x, counts):
    """
    Remove x from set_list and update counts.
    Returns (new_set_list, new_counts).
    """
    new_set = [v for v in set_list if v != x]
    new_counts = counts.copy()
    
    # Decrement counts for sums involving x
    for y in set_list:
        if y != x:
            s = x + y
            new_counts[s] -= 1
    new_counts[2 * x] -= 1
    
    # Clean up any negative values (shouldn't happen but safe)
    new_counts[new_counts < 0] = 0
    
    return new_set, new_counts

def _count_violations(set_list, counts):
    """Count how many sums have count > 2."""
    return int(np.sum(counts > 2))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Step 1: Greedy baseline
    greedy_set, greedy_counts = _greedy_build()
    current = sorted(greedy_set)
    current_set = set(current)
    current_counts = greedy_counts
    
    best = list(current)
    best_size = len(current)
    
    # Step 2: Local Search
    rng = random.Random(42)
    all_vals = list(range(1, 1001))
    
    stagnation_counter = 0
    max_stagnation = 500
    
    # Track which elements are "tight" - involved in sums with count == 2
    # We'll recompute this periodically
    
    while time.time() < deadline:
        op = rng.random()
        
        if op < 0.5:
            # Pure add
            x = rng.choice(all_vals)
            if x in current_set:
                continue
            success, new_set, new_counts = _try_add(current, x, current_counts)
            if success:
                current = new_set
                current_set = set(current)
                current_counts = new_counts
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        elif op < 0.9:
            # Swap: remove one, add one
            if not current:
                continue
            
            remove_x = rng.choice(current)
            add_x = rng.choice(all_vals)
            if add_x == remove_x:
                continue
            if add_x in current_set:
                continue
            
            # Temporarily remove remove_x
            temp_set, temp_counts = _remove(current, remove_x, current_counts)
            temp_set_set = set(temp_set)
            
            if add_x in temp_set_set:
                continue
            
            # Try to add add_x
            success, new_set, new_counts = _try_add(temp_set, add_x, temp_counts)
            if success:
                current = new_set
                current_set = set(current)
                current_counts = new_counts
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        else:
            # Remove one random element
            if current:
                remove_x = rng.choice(current)
                current, current_counts = _remove(current, remove_x, current_counts)
                current_set = set(current)
                stagnation_counter += 1
        
        # If stuck, do a kick
        if stagnation_counter > max_stagnation:
            for _ in range(3):
                if current:
                    remove_x = rng.choice(current)
                    current, current_counts = _remove(current, remove_x, current_counts)
                    current_set = set(current)
            stagnation_counter = 0
    
    return best
