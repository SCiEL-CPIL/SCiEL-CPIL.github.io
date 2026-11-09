"""Find the maximum interval in [1,1008] with no modular inverse pair (including self-inverse)."""
# EVOLVE-BLOCK-START
def construct():
    mod = 1009
    n = 1008  # domain [1, 1008]
    
    # Precompute modular inverses
    inv = [0] * (mod + 1)
    for x in range(1, mod):
        inv[x] = pow(x, mod - 2, mod)
    
    # Domain for valid intervals: [2, 1007] since 1 and 1008 are self-inverse
    # (1*1 % 1009 = 1, 1008*1008 % 1009 = 1)
    
    # For each x in [2, 1007], if inv[x] > x, then x is the "left" member of the pair
    # The constraint: for interval [a, b], if x >= a and x is left member, then inv[x] > b
    # So b < min{ inv[x] | x in [a, 1007], inv[x] > x }
    
    # right_end[x] = inv[x] if x is left member (inv[x] > x), else 1008 (infinity)
    right_end = [1008] * (mod + 1)
    
    for x in range(2, 1008):
        ix = inv[x]
        if ix > x:
            right_end[x] = ix
    
    # Compute suffix minimum: suffix_min[a] = min(right_end[a], right_end[a+1], ..., right_end[1007])
    suffix_min = [1008] * (mod + 1)
    suffix_min[1008] = 1008
    
    for a in range(1007, 1, -1):
        suffix_min[a] = min(right_end[a], suffix_min[a + 1])
    
    best_start = 2
    best_len = 0
    
    for a in range(2, 1008):
        # Maximum valid end b is suffix_min[a] - 1, capped at 1007
        b_max = min(suffix_min[a] - 1, 1007)
        
        if b_max >= a:
            length = b_max - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
