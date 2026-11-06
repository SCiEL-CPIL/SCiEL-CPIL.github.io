import time

def construct():
    p = 1009
    n = 1008
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_len = 1
    
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        # For start s, find max L such that for all x in [s, s+L-1], inv[x] not in [s, s+L-1]
        # Track the minimum inv[x] value >= s as we extend the window
        max_L = 0
        min_conflict_inv = n + 1  # minimum inv[x] for x in [s, e] where inv[x] >= s
        for e in range(s, n + 1):
            x = e
            ix = inv[x]
            if ix >= s:
                if ix < min_conflict_inv:
                    min_conflict_inv = ix
            # Interval [s, e] is valid iff no x in [s,e] has inv[x] in [s,e]
            # i.e., min_conflict_inv > e (all conflicting inverses are beyond e)
            if min_conflict_inv > e:
                max_L = e - s + 1
            else:
                break
        if max_L > best_len:
            best_len = max_L
            best_start = s
    
    return [best_start, best_len]
