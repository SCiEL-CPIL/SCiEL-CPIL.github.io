import time

def construct():
    MOD = 1009
    N = 1008
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # For each x, its "forbidden partner" is inv[x].
    # x and inv[x] cannot both be in the interval.
    # Self-inverse: x == inv[x] means x^2 ≡ 1 mod 1009, i.e., x=1 or x=1008.
    # These can never be in a valid interval.
    
    # Two-pointer sliding window:
    # Maintain window [L, R] such that no two elements in the window are modular inverses.
    # For each R, extend L as needed to avoid conflicts.
    
    best_len = 0
    best_start = 1
    
    L = 1
    # Track which elements are in the window via a set of their "pair representatives"
    # Use a set of actual values in window for O(1) inverse lookup
    window_set = set()
    
    for R in range(1, MOD):
        # R is in [1, 1008]
        if R == 1 or R == 1008:
            # Self-inverse: cannot be in window, clear everything up to and including R
            while L <= R:
                window_set.discard(L)
                L += 1
            continue
        
        # Check if inv[R] is already in the window
        r_inv = inv[R]
        while r_inv in window_set:
            window_set.discard(L)
            L += 1
        
        window_set.add(R)
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
        
        if time.time() > deadline:
            break
    
    # Verify the result
    if best_len < 1:
        return [2, 1]
    
    end = best_start + best_len - 1
    # Verify: no x,y in [best_start, end] with x*y % 1009 == 1
    valid = True
    for x in range(best_start, end + 1):
        if x * x % MOD == 1:
            valid = False
            break
        if x * inv[x] % MOD == 1 and inv[x] >= best_start and inv[x] <= end:
            # inv[x] is in the window, conflict!
            valid = False
            break
        if time.time() > deadline:
            break
    
    if not valid:
        # Fallback: find a small valid interval
        for s in range(1, MOD - 1):
            if s == 1 or s == 1008:
                continue
            # Check [s, s] is valid (single element, not self-inverse)
            if s * s % MOD != 1:
                return [s, 1]
        return [2, 1]
    
    return [best_start, best_len]
