"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses: inv[x] = x^(-1) mod p for x in [1, p-1]
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x * x % p == 1 => x == 1 or x == p-1 (1008)
    # These cannot be in any valid interval
    
    best_start = 2
    best_len = 1  # minimum: just [2,1] is valid since 2*2=4 != 1 mod 1009
    
    # For each possible start position a in [1, 1008]:
    #   Find the maximum b >= a such that for all x, y in [a, b], x*y % p != 1
    #   The constraint: for every x in [a,b], inv[x] must NOT be in [a,b]
    #   (including x=y, so x must not be self-inverse)
    
    # Strategy: for a given start a, the interval [a, b] is valid iff:
    #   1. a is not self-inverse (a != 1 and a != 1008)
    #   2. For every x in [a, b], inv[x] is not in [a, b]
    #      Equivalently: no x in [a,b] has inv[x] in [a,b]
    
    # We can extend b from a upward until we hit a blocking pair.
    # A pair (x, inv[x]) blocks the interval if both x and inv[x] are in [a, b].
    # For a fixed a, as we increase b, the first blocking event is when
    # b reaches max(x, inv[x]) for some x in [a, b] where inv[x] >= a.
    
    # More precisely: for start a, the interval [a, b] is invalid iff there exists
    # x in [a, b] such that inv[x] in [a, b].
    # The tightest constraint: b < min over all x in [a, ...] of max(x, inv[x])
    #   where inv[x] >= a and inv[x] <= ... 
    # Actually, let's think differently.
    
    # For start a, define f(a) = the smallest b >= a such that the interval [a, b] is invalid.
    # Then the max valid length starting at a is f(a) - a - 1 (if f(a) > a), else 0.
    
    # [a, b] is invalid iff there exists x in [a, b] with inv[x] in [a, b].
    # For a given a, consider all x >= a. If inv[x] >= a, then the pair (x, inv[x]) 
    # requires that b < max(x, inv[x]) to keep both out... no wait.
    # If both x and inv[x] are >= a, then to avoid having both in [a, b], we need
    # b < max(x, inv[x]).
    # But also x must be <= b and inv[x] must be <= b.
    # The interval [a, b] contains both x and inv[x] iff max(x, inv[x]) <= b and min(x, inv[x]) >= a.
    
    # So for start a, the blocking value is:
    #   b_min = min over all x >= a where inv[x] >= a of max(x, inv[x])
    # The max valid b is b_min - 1, and the length is b_min - a.
    # But we also need b_min > a for a valid nonempty interval.
    # If no such x exists, the interval can extend to 1008, length = 1008 - a + 1.
    
    # Also need to check: a itself must not be self-inverse.
    # If a is self-inverse, skip.
    
    # Let's compute this efficiently.
    # For each a from 1 to 1008:
    #   If a is self-inverse, skip.
    #   b_min = 1009 (infinity)
    #   For x from a to 1008:
    #       if inv[x] >= a:
    #           b_min = min(b_min, max(x, inv[x]))
    #       if b_min <= a: break  # can't get a valid interval
    #   If b_min > a:
    #       length = b_min - a
    #       Update best if length > best_len
    
    # This is O(n^2) worst case but n=1008 so it's about 500K operations, very fast.
    
    for a in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Skip self-inverse residues
        if a == 1 or a == n:  # n = 1008
            continue
        
        # Check if a*a % p == 1 (should only be 1 and 1008, but be safe)
        if (a * a) % p == 1:
            continue
        
        b_min = n + 1  # infinity sentinel (1009)
        
        for x in range(a, n + 1):
            ix = inv[x]
            if ix >= a:
                mx = max(x, ix)
                if mx < b_min:
                    b_min = mx
                    if b_min <= a:
                        break
        
        if b_min > a:
            length = b_min - a
            if length > best_len:
                best_len = length
                best_start = a
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
