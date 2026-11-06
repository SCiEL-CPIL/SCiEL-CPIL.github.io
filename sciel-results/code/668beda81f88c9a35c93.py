import time
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    MOD = 1009
    # Compute modular inverse for each x in [1, 1008]
    # Since 1009 is prime, inv(x) = x^(MOD-2) % MOD
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_len = 0
    
    # O(N^2) sweep
    # For each start s, find max valid end e
    # Condition for [s, e] to be valid:
    # For all x in [s, e], inv[x] is NOT in [s, e].
    # This is equivalent to:
    # Let M = min { inv[x] for x in [s, e] if inv[x] >= s }
    # Then we need e < M (or M is infinity if no such x exists)
    
    for s in range(1, MOD):
        if time.time() > deadline:
            break
        
        M = MOD + 1  # Infinity for our purposes, as inv[x] <= MOD-1
        # We iterate e from s to MOD-1
        # We maintain M as we extend e
        for e in range(s, MOD):
            # Add element e to the window [s, e]
            ix = inv[e]
            if ix >= s:
                if ix < M:
                    M = ix
            
            # Check validity: e must be strictly less than M
            if e < M:
                length = e - s + 1
                if length > best_len:
                    best_len = length
                    best_start = s
            else:
                # If e >= M, then inv[x] <= e for some x in [s, e] with inv[x] >= s.
                # Specifically, there is an x in [s, e] such that inv[x] = M <= e.
                # Also inv[x] >= s. So inv[x] is in [s, e].
                # Conflict found. Extending e further will not resolve this conflict.
                break
    
    # Ensure we return a non-empty interval
    # If best_len is 0, find any valid single element
    if best_len == 0:
        for s in range(1, MOD):
            if inv[s] != s: # Not self-inverse
                best_start = s
                best_len = 1
                break
    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
