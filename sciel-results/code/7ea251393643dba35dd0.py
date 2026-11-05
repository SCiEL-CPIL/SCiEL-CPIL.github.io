"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse elements that cannot appear in any valid interval
    # x^2 % p == 1 means x is its own inverse
    self_inverse = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inverse.add(x)
    # self_inverse = {1, 1008}
    
    best_len = 0
    best_start = 1
    
    # Try all possible intervals
    for start in range(1, n + 1):
        # If start is a self-inverse, skip
        if start in self_inverse:
            continue
        
        # Extend the interval as far as possible
        end = start
        valid = True
        # We'll track which elements are in our interval
        # For efficiency, check incrementally
        while end <= n:
            # If end is a self-inverse, we can't include it
            if end in self_inverse:
                break
            
            # Check if inv(end) is already in [start, end-1]
            inv_end = inv[end]
            if start <= inv_end <= end - 1:
                break
            
            end += 1
        
        # The valid interval is [start, end-1]
        length = end - start
        if length > best_len:
            best_len = length
            best_start = start
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
