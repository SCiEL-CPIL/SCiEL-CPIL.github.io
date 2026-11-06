import time

def construct():
    N = 51975
    # Get divisors
    divisors = [d for d in range(3, N + 1, 2) if N % d == 0]
    
    # Precompute base masks S_m
    # S_m has bits at 0, m, 2m...
    S_masks = {}
    for m in divisors:
        mask = 0
        for k in range(N // m):
            mask |= (1 << (k * m))
        S_masks[m] = mask
        
    # Precompute all shifted masks
    # Masks[m] is a list of length m
    Masks = {}
    ALL_ONES = (1 << N) - 1
    for m in divisors:
        S = S_masks[m]
        masks_for_m = []
        for r in range(m):
            if r == 0:
                mask = S
            else:
                # Cyclic shift
                # (S << r) | (S >> (N - r))
                # Need to mask to N bits? S is already < 2^N.
                # Shifts might exceed N bits?
                # S has max bit at N-m.
                # S << r has max bit at N-m+r.
                # If r < m, N-m+r < N.
                # So S << r is < 2^N.
                # S >> (N-r) moves high bits to low.
                mask = (S << r) | (S >> (N - r))
                mask &= ALL_ONES # Just in case
            masks_for_m.append(mask)
        Masks[m] = masks_for_m

    uncovered = ALL_ONES
    remaining = set(divisors)
    classes = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    while remaining:
        best_count = N + 1
        best_m = None
        best_r = None
        best_new_uncov = None
        
        # Iterate through remaining moduli
        # To allow early termination if time runs out?
        # We need to pick the best among ALL remaining.
        # If we don't finish checking all, we can't guarantee optimality of this step.
        # But we can return the best found so far if time is up.
        
        for m in remaining:
            masks_for_m = Masks[m]
            for r in range(m):
                if time.time() > deadline:
                    # Break out and return current best
                    break
                mask = masks_for_m[r]
                new_uncov = uncovered & ~mask
                c = new_uncov.bit_count()
                if c < best_count:
                    best_count = c
                    best_m = m
                    best_r = r
                    best_new_uncov = new_uncov
            if time.time() > deadline:
                break
        
        if best_m is None:
            # Timeout or no improvement?
            break
            
        uncovered = best_new_uncov
        remaining.remove(best_m)
        classes.append([best_m, best_r])
        
    return classes
