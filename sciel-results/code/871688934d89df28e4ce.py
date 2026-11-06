"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse elements: x * x % p == 1 => x == 1 or x == p-1 = 1008
    # These cannot be in any valid interval (diagonal check)
    
    # The constraint: for every x, y in [a, b] (including x=y), x*y % p != 1
    # This means:
    #   1. No x in [a,b] can be self-inverse (x=1 or x=1008)
    #   2. For no x in [a,b] is inv(x) also in [a,b]
    #      (because if both x and inv(x) are in [a,b], then x*inv(x)%p = 1)
    
    # For a given interval [a, b], it's valid iff:
    #   - a != 1 and b != 1008 (and no self-inverse in range, but only 1 and 1008 are self-inverse)
    #   - For all x in [a, b], inv(x) is not in [a, b]
    
    # Strategy: For each start a, find the maximum b such that [a,b] is valid.
    # As b increases, the constraint only gets tighter (more elements to check).
    # So for a fixed a, there's a maximum valid b.
    
    best_start = 2
    best_len = 0
    
    for a in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Skip if a is self-inverse
        if a == 1 or a == 1008:
            continue
        
        # Binary search or linear scan for max b
        # Since n=1008 is small, let's just scan
        # But we can be smarter: start from the current best and see if we can do better
        
        # For a given a, the interval [a, b] is invalid if there exists x in [a,b] 
        # such that inv(x) is also in [a,b], OR if a or b is self-inverse.
        
        # Let's find the maximum b by scanning
        max_b = a - 1
        for b in range(a, n + 1):
            if time.time() > deadline:
                break
            if b == 1 or b == 1008:
                # b is self-inverse, so [a,b] includes a self-inverse element
                # Actually we need to check if any element in [a,b] is self-inverse
                # Since a != 1 and a != 1008, and we're scanning b upward,
                # the first self-inverse we hit would be 1008 (since a >= 2)
                # Wait, 1 < a (since a != 1 and a >= 1, so a >= 2), so 1 is not in [a,b]
                # 1008 could be in [a,b] if b >= 1008
                break
            # Check if inv(b) is in [a, b]
            if inv[b] >= a and inv[b] <= b:
                break
            max_b = b
        
        length = max_b - a + 1
        if length > best_len:
            best_len = length
            best_start = a
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
