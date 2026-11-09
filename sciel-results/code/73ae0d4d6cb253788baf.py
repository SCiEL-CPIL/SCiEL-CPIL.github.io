"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    lo_bound = 2
    hi_bound = 1007
    
    # b_arr[a] will store the maximum element b of the pair {a, b} where a is the smaller element.
    # If a is not the smaller element of any pair (or has no pair within bounds), it remains infinity.
    INF = float('inf')
    b_arr = [INF] * (MOD + 2)
    
    for x in range(lo_bound, hi_bound + 1):
        y = inv[x]
        # y is guaranteed to be in [2, 1007] for x in [2, 1007]
        a = x if x < y else y
        b = y if x < y else x
        b_arr[a] = b
        
    # Compute suffix minimums
    # suffix_min[i] = min(b_arr[i], b_arr[i+1], ..., b_arr[hi_bound])
    suffix_min = [INF] * (MOD + 3)
    for i in range(hi_bound, lo_bound - 1, -1):
        suffix_min[i] = min(b_arr[i], suffix_min[i + 1])
        
    best_start = lo_bound
    best_len = 0
    
    for L in range(lo_bound, hi_bound + 1):
        min_b = suffix_min[L]
        if min_b == INF:
            max_R = hi_bound
        else:
            max_R = min_b - 1
            
        if max_R >= L:
            length = max_R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
        # Deadline check
        if time.time() - start_time > DEADLINE:
            break
            
    # Fallback if no valid interval found (should not happen)
    if best_len == 0:
        best_start = lo_bound
        best_len = 1
        
    return [best_start, best_len]

# EVOLVE-BLOCK-END
