import time

def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_len = 1
    
    for s in range(1, n + 1):
        # For start s, find max L such that for all x in [s, s+L-1], inv[x] not in [s, s+L-1]
        # Greedy: extend L as far as possible
        max_L = 0
        for L in range(1, n - s + 2):
            end = s + L - 1
            # Check if any x in [s, end] has inv[x] in [s, end]
            valid = True
            for x in range(s, end + 1):
                if s <= inv[x] <= end:
                    valid = False
                    break
            if valid:
                max_L = L
            else:
                break
        if max_L > best_len:
            best_len = max_L
            best_start = s
    
    return [best_start, best_len]
