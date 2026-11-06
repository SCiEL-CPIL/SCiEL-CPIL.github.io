import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Conservative safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Two-pointer approach:
    # We want the longest subarray [s, e] such that no pair (x, y) with x*y % MOD == 1 exists within [s, e].
    # A pair exists if there is an x in [s, e] such that inv[x] is also in [s, e].
    # Self-inverse elements (1 and 1008) cannot be in any valid interval.
    
    best_len = 0
    best_start = 1
    
    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Handle self-inverse: 1 and 1008 cannot be in any valid interval
        if e == 1 or e == 1008:
            s = e + 1
            continue
            
        # Handle distinct pair: if inv[e] is in [s, e-1], move s past it
        ie = inv[e]
        if ie >= s and ie < e:
            s = ie + 1
            
        # Now [s, e] should be valid
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    # Verify the result is valid; if not, do a simple fallback
    def is_valid(start, length):
        end = start + length - 1
        if start < 1 or end > N or length <= 0:
            return False
        for x in range(start, end + 1):
            if (x * x) % MOD == 1:
                return False
            ix = inv[x]
            if ix != x and start <= ix <= end:
                return False
        return True
    
    if best_len == 0:
        return [2, 1]
    
    if not is_valid(best_start, best_len):
        # Fallback: scan for any valid interval
        for s2 in range(2, N):
            if time.time() > deadline:
                break
            for e2 in range(s2, N + 1):
                if e2 == 1008:
                    break
                if not is_valid(s2, e2 - s2 + 1):
                    break
                if e2 - s2 + 1 > best_len:
                    best_len = e2 - s2 + 1
                    best_start = s2
    
    return [best_start, best_len]
