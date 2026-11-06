"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # For each possible start position, find the maximum valid end
    for a in range(1, n + 1):
        # Skip self-inverse residues: 1 and 1008 (since 1*1 % 1009 = 1, 1008*1008 % 1009 = 1)
        if a == 1 or a == 1008:
            continue
        
        # Find the maximum b >= a such that for all x, y in [a, b], x*y % 1009 != 1
        # The constraint is: for every x in [a,b], inv(x) must NOT be in [a,b]
        # (since if inv(x) is in [a,b], then x * inv(x) % 1009 = 1)
        # Also need to check x=x: x*x % 1009 != 1, which means x != 1 and x != 1008 (already handled)
        
        # For a fixed start a, the interval [a, b] is valid iff:
        # For all x in [a, b]: inv(x) not in [a, b]
        # This is equivalent to: min(inv(x) for x in [a,b]) < a OR max(inv(x) for x in [a,b]) > b
        # But actually we need: for ALL x in [a,b], inv(x) < a OR inv(x) > b
        
        # Better approach: for start a, extend b as far as possible
        # b can extend until some x in [a, b] has inv(x) in [a, b]
        
        # Let's think differently:
        # For a given start a, the blocking condition is:
        # There exists x in [a, b] such that a <= inv(x) <= b
        # The tightest constraint: for each x >= a, if inv(x) >= a, then b must be < inv(x)
        # So b_max = min over all x >= a where inv(x) >= a of (inv(x) - 1)
        # But also b >= a, and we need to check that a itself is valid (a*a % 1009 != 1)
        
        # Actually, let me reconsider. For interval [a, b]:
        # Valid iff for all x, y in [a, b]: x*y % 1009 != 1
        # This means for all x in [a, b]: inv(x) not in [a, b]
        # (because if y = inv(x), then x*y = 1 mod 1009)
        
        # So for start a, the maximum valid b is:
        # b = min(1008, min{inv(x) - 1 : x in [a, 1008], inv(x) >= a})
        # But we also need b >= a
        
        # Let's compute this efficiently
        # For each a, we need the minimum of inv(x) - 1 over x in [a, 1008] where inv(x) >= a
        
        # Precompute: for each position x, we know inv(x)
        # For a given a, scan x from a to 1008, track min inv(x) where inv(x) >= a
        
        max_b = 1008
        for x in range(a, 1009):
            if x == 1 or x == 1008:
                # These are self-inverse, so if they're in the interval, it's invalid
                # But we already skip a=1 and a=1008 as starts
                # If x is in [a, 1008] and x is 1 or 1008, that's a problem only if a <= 1 <= 1008 or a <= 1008 <= 1008
                # Since a >= 2 (we skip a=1), x=1 is never in [a, 1008] for a >= 2
                # x=1008 is in [a, 1008] if a <= 1008, which is always true
                # So if 1008 is in the interval, it's invalid because 1008*1008 % 1009 = 1
                # Thus b must be < 1008, i.e., b <= 1007
                if x == 1008:
                    max_b = min(max_b, 1007)
                continue
            
            ix = inv[x]
            if ix >= a:
                # Then b must be < ix, so b <= ix - 1
                max_b = min(max_b, ix - 1)
        
        if max_b >= a:
            length = max_b - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
