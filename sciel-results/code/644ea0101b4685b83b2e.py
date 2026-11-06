def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85

    p = 1009
    n = 1008

    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)

    best_start = 1
    best_length = 1

    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # We want the largest e such that [s, e] is valid.
        # We can iterate e from n down to s.
        # Optimization: 
        # If we know the max possible e is constrained by the smallest inv[x] >= s for x in [s, n],
        # we can start e from there.
        
        # Let's just iterate e downwards.
        # To speed up, we can precompute a prefix/suffix structure?
        # Or just use the boolean array check.
        
        # Since N is small, let's just check.
        # We can break early if we find a valid e.
        
        # Actually, we can compute the max valid e directly.
        # max_e = n
        # for x in range(s, n+1):
        #    if inv[x] >= s:
        #       max_e = min(max_e, inv[x] - 1)
        #       # Also if x > max_e, we don't care? 
        #       # If x > max_e, it's not in [s, max_e], so it doesn't constrain max_e directly 
        #       # unless its inv is in [s, max_e].
        #       # But if inv[x] < s, it's safe.
        #       # If inv[x] >= s, then we need inv[x] > max_e OR x > max_e.
        #       # Since we are setting max_e based on inv[x], we ensure inv[x] > max_e.
        #       # Do we need to ensure x > max_e? No, x is in [s, n]. 
        #       # If x <= max_e, then we need inv[x] > max_e.
        #       # If x > max_e, then x is not in the interval, so no constraint.
        #       # So the condition is: 
        #       # e < inv[x] for all x in [s, e] with inv[x] >= s.
        #       # This is the same as e < min{ inv[x] | x in [s, n], inv[x] >= s }.
        #       # Let M = min{ inv[x] | x in [s, n], inv[x] >= s }.
        #       # Then e can be M - 1.
        
        min_inv = n + 1
        for x in range(s, n + 1):
            if inv[x] >= s:
                if inv[x] < min_inv:
                    min_inv = inv[x]
        
        if min_inv == n + 1:
            # No constraints
            e = n
        else:
            e = min_inv - 1
            
        if e >= s:
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    return [best_start, best_length]
