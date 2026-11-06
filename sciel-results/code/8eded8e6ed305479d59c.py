import time

def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    # The interval must be within [1, 1008].
    # 1 and 1008 are self-inverses (1*1=1, 1008*1008=1).
    # So they cannot be in the interval.
    # Thus s >= 2 and e <= 1007.
    
    best_start = 2
    best_length = 0
    
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    for s in range(2, n):  # s can be up to 1007
        if time.time() > deadline:
            break
            
        max_e = n  # Start with max possible, will be reduced
        
        # We need e such that for all x in [s, e], inv[x] is not in [s, e].
        # If inv[x] < s, it's fine.
        # If inv[x] >= s, then we must have e < inv[x] (since x is in [s, e], 
        # if inv[x] is also in [s, e], it's a conflict).
        # Actually, if inv[x] >= s, the conflict pair is {x, inv[x]}.
        # The interval [s, e] contains this pair if e >= max(x, inv[x]).
        # So we need e < max(x, inv[x]).
        
        for x in range(s, n + 1):
            if time.time() > deadline:
                break
            ix = inv[x]
            if ix >= s:
                # The pair (x, ix) is "active" for start s.
                # The interval ends before the larger of the two.
                limit = max(x, ix)
                if limit - 1 < max_e:
                    max_e = limit - 1
                    
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    return [best_start, best_length]
