import time
import random
import sys
import math
import numpy as np

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1

def _get_counts(A_set):
    """
    Returns a list of representation counts for sums in [_LO, _HI].
    Returns None if A is empty.
    """
    if not A_set:
        return None
    A_list = list(A_set)
    A_arr = np.array(A_list, dtype=np.int32)
    n = len(A_arr)
    
    if n == 0:
        return [0] * _RANGE

    # Compute all pair sums
    sums = A_arr[:, None] + A_arr[None, :]
    
    # Get upper triangle indices (i <= j)
    i, j = np.triu_indices(n)
    pair_sums = sums[i, j]
    
    # Filter to target range
    mask = (pair_sums >= _LO) & (pair_sums <= _HI)
    valid_sums = pair_sums[mask]
    
    if len(valid_sums) == 0:
        return [0] * _RANGE
        
    # Count
    counts = np.bincount(valid_sums - _LO, minlength=_RANGE)
    return counts.tolist()

def _is_valid(counts):
    """Check if all sums in the range are covered."""
    if counts is None:
        return False
    return all(c >= 1 for c in counts)

def _get_peak(counts):
    """Get the maximum representation count."""
    if counts is None:
        return float('inf')
    return max(counts)

def _cost(A_set):
    """
    Cost function for Simulated Annealing.
    Penalizes missing sums heavily, then minimizes peak.
    """
    counts = _get_counts(A_set)
    if counts is None:
        return float('inf')
    
    missing = 0
    peak = 0
    for c in counts:
        if c == 0:
            missing += 1
        elif c > peak:
            peak = c
            
    return peak + 1000 * missing

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    rng = random.Random(42)
    
    best_A = None
    best_cost = float('inf')
    best_peak = float('inf')
    
    # Generate 10 random candidates to find a good start
    for _ in range(10):
        size = rng.randint(50, 70)
        A_cand = set(rng.sample(range(_N + 1), size))
        c = _cost(A_cand)
        if c < best_cost:
            best_cost = c
            best_A = A_cand.copy()
            if c < 1000:
                best_peak = c

    if best_A is None:
        return list(range(_N + 1))
        
    current_A = best_A.copy()
    current_cost = best_cost
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    last_improve_time = start_time
    
    while time.time() < DEADLINE:
        # Generate a move
        move_type = rng.random()
        new_A = current_A.copy()
        
        if move_type < 0.8:
            # Swap
            if len(current_A) == 0:
                continue
            in_A = list(current_A)
            x = in_A[rng.randint(0, len(in_A) - 1)]
            not_in_A = [x for x in range(_N + 1) if x not in current_A]
            if not not_in_A:
                continue
            y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
            new_A.discard(x)
            new_A.add(y)
        elif move_type < 0.9:
            # Add
            not_in_A = [x for x in range(_N + 1) if x not in current_A]
            if not not_in_A:
                continue
            y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
            new_A.add(y)
        else:
            # Remove
            if len(current_A) == 0:
                continue
            in_A = list(current_A)
            x = in_A[rng.randint(0, len(in_A) - 1)]
            new_A.discard(x)
            
        if not new_A:
            continue
            
        new_cost = _cost(new_A)
        delta = new_cost - current_cost
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_A = new_A
            current_cost = new_cost
            
            # Check if this is a new best valid solution
            if current_cost < 1000 and current_cost < best_peak:
                best_peak = current_cost
                best_A = current_A.copy()
                last_improve_time = time.time()
                
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Restart if stuck
        if time.time() - last_improve_time > 5.0:
            current_A = best_A.copy()
            current_cost = best_cost
            T = 5.0
            last_improve_time = time.time()

    if best_peak == float('inf'):
        return list(range(_N + 1))
        
    return sorted(best_A)
