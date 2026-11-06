import time

def construct():
    p = 1009
    n = 1008
    deadline = time.time() + 85.0
    
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_len = 0
    
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        max_L = 0
        min_conflict_inv = n + 1
        for e in range(s, n + 1):
            ix = inv[e]
            if ix >= s:
                if ix < min_conflict_inv:
                    min_conflict_inv = ix
            if min_conflict_inv > e:
                max_L = e - s + 1
            else:
                break
        if max_L > best_len:
            best_len = max_L
            best_start = s
            
    if best_len == 0:
        # Fallback if no valid interval found (should not happen)
        return [1, 1]
        
    return [best_start, best_len]
