import time
import random
import sys

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1

def _get_counts(A_set):
    if not A_set:
        return None
    A_list = sorted(A_set)
    n = len(A_list)
    counts = [0] * _RANGE
    for i in range(n):
        a = A_list[i]
        for j in range(i, n):
            b = A_list[j]
            s = a + b
            if _LO <= s <= _HI:
                counts[s - _LO] += 1
    return counts

def _is_valid(counts):
    if counts is None:
        return False
    return all(c >= 1 for c in counts)

def _get_objective(counts):
    if counts is None:
        return (float('inf'), float('inf'))
    peak = max(counts)
    sum_sq = sum(c * c for c in counts)
    return (peak, sum_sq)

def _simulate_swap(A_set, counts, x, y):
    new_counts = counts[:]
    
    # Remove x
    for a in A_set:
        s = x + a
        if _LO <= s <= _HI:
            idx = s - _LO
            new_counts[idx] -= 1
            if new_counts[idx] < 0:
                return (float('inf'), float('inf'), False, None)
                
    # Add y
    # Pairs {y, a} for a in A_set where a != x
    for a in A_set:
        if a == x:
            continue
        s = y + a
        if _LO <= s <= _HI:
            idx = s - _LO
            new_counts[idx] += 1
            
    s_2y = 2 * y
    if _LO <= s_2y <= _HI:
        idx = s_2y - _LO
        new_counts[idx] += 1
        
    if not _is_valid(new_counts):
        return (float('inf'), float('inf'), False, None)
        
    peak = max(new_counts)
    sum_sq = sum(c * c for c in new_counts)
    return (peak, sum_sq, True, new_counts)

def construct():
    start_time = time.time()
    # Hard limit 60s, safety margin 50s
    DEADLINE = start_time + 50.0
    
    # Start with full set
    A = set(range(0, _N + 1))
    counts = _get_counts(A)
    if not counts:
        return [0]
        
    # Initialize best
    best_A = A.copy()
    best_counts = counts[:]
    best_obj = _get_objective(counts)
    curr_A = A.copy()
    curr_counts = counts[:]
    curr_obj = best_obj
    
    # Candidates for y (elements not in A)
    candidates = [i for i in range(_N + 1) if i not in curr_A]
    # Initially A is full, so candidates is empty. 
    # We need to start by removing elements or swapping with "outside" is impossible.
    # So we must first reduce the set size? 
    # If A is full, we cannot add any y. We can only remove.
    # But the task asks for a subset. 
    # If we start with full set, we are at a local minimum for size, but not for multiplicity.
    # We cannot swap if there are no candidates.
    # We must allow removal? Or start with a smaller set.
    
    # Let's start with a random subset of size 80 to have candidates.
    # Or, better: Start with full set, but allow "removal" swaps? 
    # The interface is swap x for y. If y is not in [0,128], it's a removal.
    # But the problem says A is a subset of [0,128].
    # So we should start with a set that has "holes".
    
    # Strategy: 
    # 1. Create a random subset of size 100.
    # 2. Run local search.
    
    rng = random.Random(42)
    
    # Generate initial random subset
    # We want to cover [128, 256]. 
    # A random set of size 100 from 129 elements is likely to cover it.
    # Let's try to ensure validity.
    
    attempts = 0
    while time.time() < DEADLINE - 5 and attempts < 10:
        # Pick 100 random elements
        init_A = set(rng.sample(range(_N + 1), 100))
        init_counts = _get_counts(init_A)
        if init_counts and _is_valid(init_counts):
            curr_A = init_A
            curr_counts = init_counts
            curr_obj = _get_objective(curr_counts)
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_A = curr_A.copy()
                best_counts = curr_counts[:]
            break
        attempts += 1
        
    if not _is_valid(curr_counts):
        # Fallback to full set if we couldn't find a valid random start
        curr_A = A.copy()
        curr_counts = counts[:]
        curr_obj = best_obj
        if curr_obj < best_obj:
            best_obj = curr_obj
            best_A = curr_A.copy()
            best_counts = curr_counts[:]

    # Local Search
    # Re-initialize candidates based on curr_A
    candidates = [i for i in range(_N + 1) if i not in curr_A]
    
    while time.time() < DEADLINE:
        if not candidates:
            break
            
        # Pick x from A
        x = rng.choice(list(curr_A))
        
        # Pick y from candidates
        y_idx = rng.randint(0, len(candidates) - 1)
        y = candidates[y_idx]
        
        # Simulate swap
        new_peak, new_sum_sq, is_valid, new_counts = _simulate_swap(curr_A, curr_counts, x, y)
        
        if is_valid:
            new_obj = (new_peak, new_sum_sq)
            if new_obj < curr_obj:
                # Accept
                curr_A.discard(x)
                curr_A.add(y)
                
                # Update candidates
                # Replace y with x in candidates list
                for i, val in enumerate(candidates):
                    if val == y:
                        candidates[i] = x
                        break
                        
                curr_counts = new_counts
                curr_obj = new_obj
                
                if curr_obj < best_obj:
                    best_obj = curr_obj
                    best_A = curr_A.copy()
                    best_counts = curr_counts[:]
            else:
                pass
        else:
            pass
            
    return sorted(best_A)
