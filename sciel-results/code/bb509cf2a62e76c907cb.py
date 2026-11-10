import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    # inv[x] is the modular inverse of x mod 1009
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # Self-inverse elements are those where inv[x] == x
    # These are 1 and 1008
    # We can check this on the fly or precompute a boolean array
    is_self_inverse = [False] * (MOD + 1)
    for x in range(1, MOD):
        if inv[x] == x:
            is_self_inverse[x] = True

    best_start = 2
    best_len = 0
    
    e = 0 # Current right end of the window, 0-indexed logic but using 1-based values
    # We iterate start s from 1 to 1008
    for s in range(1, MOD):
        if time.time() > deadline:
            break
            
        # Ensure e is at least s-1
        if e < s - 1:
            e = s - 1
            
        # Extend e as far as possible
        while e + 1 < MOD:
            candidate = e + 1
            
            # If candidate is self-inverse, it cannot be in the interval
            if is_self_inverse[candidate]:
                break
                
            # Check if inv[candidate] is in the current window [s, e]
            # If it is, then adding candidate creates a pair {candidate, inv[candidate]}
            # Note: inv[candidate] != candidate because we checked is_self_inverse
            if inv[candidate] >= s and inv[candidate] <= e:
                break
                
            e = candidate
            
        # Now window is [s, e]
        if e >= s:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    # Fallback
    if best_len == 0:
        best_start = 2
        best_len = 1
        
    return [best_start, best_len]
