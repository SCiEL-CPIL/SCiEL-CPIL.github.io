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
                return (float('inf'), float('inf'), None)

    # Add y
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
        return (float('inf'), float('inf'), None)
        
    peak = max(new_counts)
    sum_sq = sum(c * c for c in new_counts)
    return (peak, sum_sq, new_counts)

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    A = set(range(0, _N + 1))
    A_list = list(A)
    counts = _get_counts(A)
    curr_obj = _get_objective(counts)
    
    best_A = A.copy()
    best_obj = curr_obj
    
    rng = random.Random(42)
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < DEADLINE:
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            max_no_improve += 1000
            
        x_idx = rng.randint(0, len(A_list) - 1)
        x = A_list[x_idx]
        
        y = rng.randint(0, _N)
        # Check if y in A
        # Using set membership is O(1)
        while y in A:
            y = rng.randint(0, _N)
            
        new_peak, new_sum_sq, new_counts = _simulate_swap(A, counts, x, y)
        
        if new_counts is not None:
            new_obj = (new_peak, new_sum_sq)
            
            # Accept if better
            # Or accept if same peak and sum_sq is not much worse (exploration)
            # Let's stick to strict improvement for now to ensure convergence.
            # If stuck, the "kick" logic increases max_no_improve, allowing more random tries.
            
            if new_obj < curr_obj:
                A.discard(x)
                A.add(y)
                
                # Update A_list
                A_list[x_idx] = y
                # A_list is no longer sorted, but we don't need it sorted for rng.choice
                # We just need it to reflect A.
                
                counts = new_counts
                curr_obj = new_obj
                no_improve_count = 0
                
                if curr_obj < best_obj:
                    best_obj = curr_obj
                    best_A = A.copy()
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    return sorted(best_A)
