import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    p = 1009
    N = p - 1  # 1008
    
    # Precompute modular inverses using Fermat's little theorem (p is prime)
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    # For each start s, find the maximum end e such that [s, e] is valid
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is NOT in [s, e]
    # i.e., inv[x] < s or inv[x] > e
    
    for s in range(1, p):
        if time.time() > deadline - 2.0:
            break
        
        # For fixed s, find max e such that no x in [s, e] has inv[x] in [s, e]
        # We can iterate e from s to p-1 and check validity incrementally
        # But we can be smarter: track the minimum "blocking" inverse
        
        # An element x in [s, e] blocks the interval if s <= inv[x] <= e
        # So for fixed s, we need e < min{inv[x] | x in [s, e] and inv[x] >= s}
        
        # Let's iterate e and maintain whether the interval is still valid
        valid = True
        for e in range(s, p):
            # Check if adding e creates a conflict
            # e conflicts with some x in [s, e-1] if inv[e] is in [s, e-1]
            # Or e is self-inverse (inv[e] == e)
            
            if inv[e] >= s and inv[e] <= e:
                # Conflict: either inv[e] is a previous element in [s, e-1] (if inv[e] < e)
                # or inv[e] == e (self-conflict)
                valid = False
                break
            else:
                # No conflict with e, interval [s, e] is valid so far
                length = e - s + 1
                if length > best_len:
                    best_len = length
                    best_start = s
    
    return [best_start, best_len]
