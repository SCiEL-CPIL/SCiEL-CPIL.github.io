import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [start, start+length-1] within [1, 1008]
    such that for all x, y in the interval (including x=y), x*y % 1009 != 1.
    
    Since 1009 is prime, every x in [1, 1008] has a unique inverse mod 1009.
    The constraint is violated iff y = x^{-1} mod 1009 for some x, y in the interval.
    Self-inverses: x = x^{-1} means x^2 = 1 mod 1009, giving x = 1 and x = 1008.
    These cannot appear in any valid interval.
    
    We use a two-pointer / sliding window approach to find the maximum-length
    valid interval in O(n) time after O(n) inverse precomputation.
    """
    p = 1009
    n = 1008
    
    # Deadline guard with safety margin
    start_time = time.time()
    deadline = start_time + 50
    
    # Precompute modular inverses for all x in [1, 1008]
    # Using the linear time formula: inv[i] = (p - (p//i) * inv[p%i] % p) % p
    inv = [0] * (p + 1)
    inv[1] = 1
    for i in range(2, p):
        inv[i] = (p - (p // i) * inv[p % i] % p) % p
    
    best_len = 0
    best_start = 1
    L = 1  # Left endpoint of current valid window
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Self-inverses: 1*1 % 1009 == 1 and 1008*1008 % 1009 == 1
        # These elements cannot be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is within the current window [L, R-1], we must advance L
        # past invR to exclude the pair (invR, R)
        if L <= invR < R:
            L = invR + 1
        
        if L <= R:
            curr_len = R - L + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = L
    
    # Safety: ensure we return a valid non-empty interval
    if best_len == 0:
        # Fallback: find any valid single element
        for x in range(2, 1008):
            if x * x % p != 1:
                best_start = x
                best_len = 1
                break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
