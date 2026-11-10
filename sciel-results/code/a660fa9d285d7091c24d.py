import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    MOD = 1009
    n = MOD - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_len = 0
    
    right = 0  # window is [left, right], initially empty
    
    for left in range(1, MOD):
        if time.time() - start_time > DEADLINE:
            break
        
        if right < left - 1:
            right = left - 1
        
        # Extend right as far as possible while maintaining validity
        while right + 1 < MOD:
            if time.time() - start_time > DEADLINE:
                break
            candidate = right + 1
            # Check if inv[candidate] is in [left, candidate]
            if left <= inv[candidate] <= candidate:
                break
            right = candidate
        
        length = right - left + 1
        if length > best_len:
            best_len = length
            best_start = left
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
