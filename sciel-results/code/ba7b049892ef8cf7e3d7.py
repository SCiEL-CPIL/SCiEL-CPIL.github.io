import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_start = 1
    best_length = 0
    best_end = 0
    
    # Sweep over all possible start positions L
    for L in range(1, N + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        # Expand R from L upwards
        R = L - 1
        while R < N:
            y = R + 1
            # The new element is y.
            # The interval would be [L, y].
            # It is valid iff inv[y] is NOT in [L, y].
            if L <= inv[y] <= y:
                break
            R = y
            
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
            best_end = R
            
    return [best_start, best_length]
