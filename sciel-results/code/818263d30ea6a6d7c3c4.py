import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Two-pointer approach:
    # We want the longest subarray [s, e] such that no pair (x, y) with x*y % MOD == 1 exists within [s, e].
    # A pair exists if there is an x in [s, e] such that inv[x] is also in [s, e].
    
    best_len = 0
    best_start = 1
    
    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Handle self-inverse
        if e == 1 or e == 1008:
            s = e + 1
            continue
            
        # Handle distinct pair
        ie = inv[e]
        if ie >= s and ie < e:
            s = ie + 1
            
        # Now [s, e] should be valid
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    # Verify the result is valid
    def is_valid_interval(start, length):
        end = start + length - 1
        if start < 1 or end > N or length <= 0:
            return False
        for x in range(start, end + 1):
            for y in range(x, end + 1):
                if (x * y) % MOD == 1:
                    return False
        return True
    
    if best_len == 0 or not is_valid_interval(best_start, best_len):
        # Fallback: find any valid interval
        for i in range(2, N + 1):
            if i != 1008:
                if is_valid_interval(i, 1):
                    return [i, 1]
        return [2, 1]
        
    return [best_start, best_len]
