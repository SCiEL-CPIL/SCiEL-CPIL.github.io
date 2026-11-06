import time

def construct():
    MOD = 1009
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_len = 0
    best_start = 2
    right = 0
    
    for left in range(1, MOD):
        if left == 1 or left == 1008:
            continue
        
        if right < left - 1:
            right = left - 1
            
        while right + 1 < MOD:
            candidate = right + 1
            if candidate == 1 or candidate == 1008:
                break
            # Check if candidate conflicts with [left, right]
            # Conflict if inv[candidate] is in [left, candidate]
            if left <= inv[candidate] <= candidate:
                break
            right = candidate
            
        length = right - left + 1
        if length > best_len:
            best_len = length
            best_start = left
            
    return [best_start, best_len]
