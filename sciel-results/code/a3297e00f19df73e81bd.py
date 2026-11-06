"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    p = 1009
    n = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # Iterate over all possible start positions
    for a in range(1, n + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        # Self-inverse check: 1 and 1008 (1008 is -1 mod 1009)
        # If a is 1 or 1008, the interval [a, a] is invalid because a*a % p == 1
        if a == 1 or a == 1008:
            continue
            
        # Check if a itself forms an inverse pair with itself (only 1 and 1008 do, already skipped)
        # But we also need to ensure inv(a) is not in [a, a], i.e., inv(a) != a
        if inv[a] == a:
            continue
            
        # Try to extend the interval to the right
        b = a
        # We can stop extending if we hit a time limit, but the loop is fast
        for k in range(a + 1, n + 1):
            if time.time() > deadline:
                break
                
            # If k is 1 or 1008, it cannot be in a valid interval
            # (Note: k > a >= 1, so k != 1. k can be 1008)
            if k == 1008:
                break
                
            # Check if adding k to [a, k-1] creates a forbidden pair
            # The new pairs formed are (k, x) for x in [a, k-1] and (k, k)
            # (k, k) is forbidden if inv[k] == k (i.e., k^2 == 1)
            # (k, x) is forbidden if x == inv[k]
            # So we need inv[k] not in [a, k]
            
            if inv[k] >= a and inv[k] <= k:
                # inv[k] is in the interval [a, k], so adding k makes it invalid
                # And any further extension will also be invalid
                break
            else:
                # Safe to add k
                b = k
                
        current_len = b - a + 1
        if current_len > best_len:
            best_len = current_len
            best_start = a
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
