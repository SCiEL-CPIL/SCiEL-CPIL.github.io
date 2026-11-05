import time

def construct():
    MOD = 1009
    N = 1008
    T = 89 # Safety margin for 90s limit, though this is O(N)

    # Precompute modular inverses
    # inv[x] is the modular inverse of x modulo MOD
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # The valid interval must be a subset of [2, 1007] because 1 and 1008 are self-inverse.
    # Let's work with the range [2, 1007].
    
    # in_window[i] will be True if i is in the current window [L, R]
    in_window = [False] * (MOD + 1)
    
    L = 2
    R = 1 # Initially empty window
    
    best_len = 0
    best_start = 2
    
    t0 = time.time()
    
    # We iterate L from 2 to 1007
    # R will only increase
    while L <= 1007:
        # Check time
        if time.time() - t0 > T:
            break
            
        # Extend R as far as possible
        # The window is [L, R]. We try to add R+1.
        # The condition for adding cand = R+1 is: inv[cand] is NOT in [L, R].
        # Since we maintain in_window for [L, R], we just check in_window[inv[cand]].
        while R < 1007:
            cand = R + 1
            # If the inverse of cand is not in the current window, we can add cand
            if not in_window[inv[cand]]:
                in_window[cand] = True
                R = cand
            else:
                break
                
        # Now [L, R] is the maximal valid interval starting at L
        cur_len = R - L + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = L
            
        # Move L forward
        in_window[L] = False
        L += 1
        
    return [best_start, best_len]
