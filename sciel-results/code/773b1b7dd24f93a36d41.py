import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    best_start = 1
    best_length = 0
    
    # Two-pointer approach:
    # For start s, extend end e as far as possible.
    # As s increases, e never needs to decrease (monotonicity).
    e = 0
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Ensure e >= s
        if e < s:
            e = s
        
        # Try to extend e as far as possible
        while e < N:
            next_e = e + 1
            # Check if adding next_e keeps [s, next_e] valid
            # Need: inv[next_e] not in [s, next_e]
            if s <= inv[next_e] <= next_e:
                break
            e = next_e
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    if best_length == 0:
        best_start = 2
        best_length = 1
    
    return [best_start, best_length]
