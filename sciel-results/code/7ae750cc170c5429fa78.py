import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_len = 0
    
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Check if L is self-inverse
        if (L * L) % MOD == 1:
            continue
            
        R = L
        
        # Expand R as far as possible
        while R < N:
            next_R = R + 1
            next_inv = inv[next_R]
            
            # Condition for [L, next_R] to be valid given [L, R] is valid:
            # next_inv must not be in [L, next_R]
            if L <= next_inv <= next_R:
                break
            R = next_R
            
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L

    return [best_start, best_len]
