"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses: inv[x] = y such that x*y % p == 1
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Find the longest valid interval [start, start+length-1] within [1, 1008]
    # where for all x, y in the interval (including x=y), x*y % 1009 != 1
    
    # Key insight: x*y % p == 1 iff y == inv[x]
    # So we need an interval where no element is the inverse of another element in the interval
    # (including itself, so no self-inverse elements: 1 and 1008)
    
    # Use a sliding window approach
    # For each right endpoint R, find the smallest L such that [L, R] is valid
    # An interval [L, R] is valid if for all x in [L, R], inv[x] is not in [L, R]
    
    best_len = 0
    best_start = 1
    
    L = 1
    for R in range(1, n + 1):
        # Self-inverse elements (1 and 1008) cannot be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in [L, R-1], we have a conflict
        # We need to move L past invR
        if L <= invR < R:
            L = invR + 1
        
        # Now [L, R] should be valid (by construction, we've moved L past all conflicts)
        # But we also need to check that no element in [L, R] has its inverse in [L, R]
        # Actually, the sliding window approach above handles this:
        # When we add R, if inv[R] is in [L, R-1], we move L to inv[R]+1
        # This ensures that for the new R, there's no conflict
        # But we need to verify that the entire window [L, R] is valid
        
        # Let's verify: for each x in [L, R], inv[x] should not be in [L, R]
        # The sliding window ensures this by construction because:
        # - When we added x (x <= R), if inv[x] was in [L_at_that_time, x-1], we moved L
        # - Since L only increases, inv[x] is not in [L, R] for any x in [L, R]
        
        current_len = R - L + 1
        if current_len > 0 and current_len > best_len:
            best_len = current_len
            best_start = L
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Also try starting from different initial L values to be safe
    # Actually, the sliding window above should find the optimal solution
    # Let's verify with a brute force check for small cases
    
    if best_len == 0:
        # Fallback: find any valid interval
        for start in range(1, n + 1):
            if start == 1 or start == 1008:
                continue
            # Check if start can be in a valid interval
            valid = True
            for x in range(start, min(start + 10, n + 1)):
                if x == 1 or x == 1008:
                    valid = False
                    break
                if inv[x] >= start and inv[x] <= min(start + 9, n):
                    valid = False
                    break
            if valid:
                best_start = start
                best_len = 1
                break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
