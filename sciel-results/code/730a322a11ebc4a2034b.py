"""Find the longest interval in [1,1008] with no modular inverse pair."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses for all x in [1, p-1]
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x such that x*x % p == 1
    # These are x=1 and x=p-1=1008
    self_inv = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inv.add(x)
    
    # For an interval [a, b] to be valid:
    # 1. No x in [a,b] is self-inverse (i.e., x != 1 and x != 1008)
    # 2. For every x in [a,b], inv[x] is NOT in [a,b]
    #    (If inv[x] is in [a,b], then x * inv[x] % p = 1, violating the constraint)
    
    best_start = 2
    best_len = 0
    
    # For each possible start position a, find the maximum valid end b
    for a in range(1, n + 1):
        # If a is self-inverse, no interval starting at a can be valid
        if a in self_inv:
            continue
        
        # Extend b as far as possible
        b = a
        while b <= n:
            # Check if b is self-inverse
            if b in self_inv:
                break
            # Check if inv[b] is in [a, b]
            ib = inv[b]
            if a <= ib <= b:
                break
            b += 1
        
        # The valid interval is [a, b-1]
        length = b - 1 - a + 1
        if length > best_len:
            best_len = length
            best_start = a
    
    # Verify the result
    a = best_start
    b = a + best_len - 1
    for x in range(a, b + 1):
        if x in self_inv:
            return [2, 30]  # fallback
        if a <= inv[x] <= b:
            return [2, 30]  # fallback
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
