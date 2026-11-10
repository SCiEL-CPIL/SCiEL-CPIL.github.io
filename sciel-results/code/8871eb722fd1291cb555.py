"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MOD = 1009
    MAX_VAL = MOD - 1  # 1008
    MAX_SEARCH = MAX_VAL - 1  # 1007, since 1008 is self-inverse
    
    # Precompute constraints
    # max_right[x]: max e such that e < inv[x] if inv[x] > x, else MAX_VAL
    # min_left[x]: min s such that s > inv[x] if inv[x] < x, else 1
    max_right = np.full(MAX_VAL + 1, MAX_VAL, dtype=int)
    min_left = np.ones(MAX_VAL + 1, dtype=int)
    
    for x in range(2, MAX_SEARCH + 1):
        ix = pow(x, MOD - 2, MOD)
        if ix > x:
            max_right[x] = ix - 1
        elif ix < x:
            min_left[x] = ix + 1
        # if ix == x (1, 1008), excluded by range
    
    best_start = 2
    best_length = 0
    
    # Sliding window search
    for s in range(2, MAX_SEARCH + 1):
        if time.time() > deadline:
            break
        
        # Early termination: if remaining possible length <= best_length, break
        # Max possible length starting at s is MAX_SEARCH - s + 1
        if MAX_SEARCH - s + 1 <= best_length:
            break
        
        # Check if s itself is valid
        if s < min_left[s]:
            continue
        
        e = s
        # Track the minimum max_right constraint in the current window [s, e]
        min_mr = max_right[s]
        
        # Extend e to the right
        while e < MAX_SEARCH:
            next_e = e + 1
            # Check if next_e can be included in [s, e]
            # 1. next_e <= min_mr (satisfies right constraints of all existing elements)
            # 2. s >= min_left[next_e] (satisfies left constraint of the new element)
            if next_e <= min_mr and s >= min_left[next_e]:
                e = next_e
                if max_right[e] < min_mr:
                    min_mr = max_right[e]
            else:
                break
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    # Fallback
    if best_length == 0:
        best_start = 2
        best_length = 1
        
    # Verification using NumPy
    end = best_start + best_length - 1
    interval = np.arange(best_start, end + 1)
    valid = True
    # Check all pairs
    for x in interval:
        for y in interval:
            if (x * y) % MOD == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Return a known safe small interval
        return [2, 1]
        
    return [best_start, best_length]
# EVOLVE-BLOCK-END
