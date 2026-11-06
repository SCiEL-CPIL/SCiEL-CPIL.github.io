"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 85 seconds, leaving 5s safety margin
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # For each possible start position, find the maximum valid end
    for a in range(2, n + 1):  # Skip a=1 (self-inverse)
        # Check if a itself is self-inverse
        if a * a % p == 1:
            continue
        
        # For interval [a, b] to be valid, for all x in [a, b]: inv(x) not in [a, b]
        # This means: for all x in [a, b], inv(x) < a OR inv(x) > b
        # The tightest constraint: b < min{inv(x) : x in [a, b], inv(x) >= a}
        # So max_b = min over x in [a, 1008] where inv(x) >= a of (inv(x) - 1)
        # Also b <= 1007 since 1008 is self-inverse
        
        max_b = 1007  # 1008 is self-inverse, so b <= 1007
        
        for x in range(a, 1008):  # Don't include 1008 (self-inverse)
            ix = inv[x]
            if ix >= a:
                # b must be < ix, so b <= ix - 1
                if ix - 1 < max_b:
                    max_b = ix - 1
                    if max_b < a:
                        break  # No valid interval starting at a
        
        if max_b >= a:
            length = max_b - a + 1
            if length > best_len:
                best_len = length
                best_start = a
        
        # Deadline check
        if time.time() > DEADLINE:
            break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
