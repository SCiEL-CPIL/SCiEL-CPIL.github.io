import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0  # Safety margin
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    # x * inv[x] % MOD == 1
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        # Fermat's little theorem: x^(p-2) % p
        inv[x] = pow(x, MOD - 2, MOD)
        
    max_len = 0
    best_start = 1
    
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
        seen = set()
        for R in range(L, N + 1):
            if time.time() > deadline:
                break
            # Check if adding R creates a conflict
            # Conflict if inv[R] is already in the window [L, R-1]
            # Or if inv[R] == R (self-inverse)
            
            # Note: inv[R] is in [1, 1008]
            if inv[R] in seen:
                break
            if inv[R] == R:
                break
            
            seen.add(R)
            current_len = R - L + 1
            if current_len > max_len:
                max_len = current_len
                best_start = L
                
    return [best_start, max_len]
