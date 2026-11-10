import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    
    # Precompute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # Precompute next blocker for each start s
    # next_blocker[s] = smallest y > s such that s <= inv[y] <= y
    # If no such y, next_blocker[s] = p (infinity)
    next_blocker = [p] * (p + 1)
    
    # We iterate y from 1 to p-1
    # For a fixed y, it is a blocker for s if:
    # 1. s < y
    # 2. s <= inv[y]
    # So s must be in [1, min(y-1, inv[y])]
    # Since we iterate y in increasing order, the first y that covers s is the min y.
    
    for y in range(1, p):
        # Calculate max s for which y is a blocker
        max_s = min(y - 1, inv[y])
        if max_s < 1:
            continue
        # For all s in [1, max_s], if next_blocker[s] is not set, set it to y
        # We can optimize this by only updating if next_blocker[s] is p
        # But since we go y increasing, the first assignment is the min.
        # We can just loop s from 1 to max_s.
        for s in range(1, max_s + 1):
            if next_blocker[s] == p:
                next_blocker[s] = y
    
    # Find max length
    best_len = 0
    best_start = 2
    
    for s in range(1, p):
        if time.time() > deadline:
            break
            
        # Check if s itself is self-inverse
        if s * s % p == 1:
            continue
            
        # Determine max end e
        # Blocker from inverse pairs: next_blocker[s] - 1
        # Blocker from self-inverse: 
        # Self inverses are 1 and 1008.
        # If s < 1008, 1008 is a potential self-inverse blocker.
        # If s == 1008, no self-inverse > s.
        # If s > 1008, not in range.
        
        e_inv = next_blocker[s] - 1
        
        e_self = 1007 if s < 1008 else 1008
        # Wait, if s < 1008, the interval can go up to 1007 because 1008 is self-inverse.
        # If s == 1008, s is self-inverse, so we skip s.
        # So for valid s, s != 1, 1008.
        # If s < 1008, max e due to self-inverse is 1007.
        # If s > 1008, not possible.
        
        e_max = min(e_inv, e_self, 1008)
        
        # Ensure e_max >= s
        if e_max < s:
            continue
            
        length = e_max - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
