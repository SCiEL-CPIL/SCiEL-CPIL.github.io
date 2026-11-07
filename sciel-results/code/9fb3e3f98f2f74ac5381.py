import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors to determine the period of the pattern
    import math
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)
        
    # We only need to search one full period
    search_limit = lcm_val
    
    best_range = float('inf')
    best_x = None
    best_L = None
    
    start_time = time.time()
    time_limit = 45.0 # Safety margin
    
    for L in range(1, search_limit + 1):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Calculate initial candidates
        candidates = []
        for i, d in enumerate(divisors):
            # Smallest multiple of d >= L
            c = ((L + d - 1) // d) * d
            candidates.append((c, d, i))
            
        # Sort by candidate value, then by divisor descending
        # This helps resolve collisions optimally for minimizing max
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        current_x = [0] * n
        next_allowed = L
        
        for c, d, i in candidates:
            if c < next_allowed:
                # Find next multiple of d >= next_allowed
                k = (next_allowed + d - 1) // d
                x = k * d
            else:
                x = c
                
            current_x[i] = x
            next_allowed = x + 1
            
        current_range = max(current_x) - L
        
        if current_range < best_range:
            best_range = current_range
            best_x = current_x[:]
            best_L = L

    return best_x
