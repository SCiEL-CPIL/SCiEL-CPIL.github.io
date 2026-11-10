import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_start = 1
    best_length = 0
    
    R = 0
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Expand R as far as possible
        while R < N:
            next_val = R + 1
            # Check if adding next_val creates a conflict in [L, R+1]
            # A conflict exists if inv[next_val] is in [L, R+1]
            # Note: inv[next_val] is in 1..1008
            if L <= inv[next_val] <= next_val:
                break
            R += 1
            
        # Current valid interval is [L, R]
        if R >= L:
            length = R - L + 1
            if length > best_length:
                best_length = length
                best_start = L
    
    if best_length == 0:
        # Fallback
        best_start = 2
        best_length = 1
        
    return [best_start, best_length]
