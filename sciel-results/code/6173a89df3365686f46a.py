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
    
    # Self-inverse elements: x^2 ≡ 1 mod p => x ≡ 1 or x ≡ p-1
    # So 1 and 1008 are self-inverse and cannot appear in any valid interval.
    
    # Two-pointer approach:
    # We want the longest contiguous subarray [s, e] ⊆ [1, N] such that
    # for every x, y in [s, e], x*y % MOD != 1.
    # Key insight: when extending the window to include e, the only new
    # conflict is between e and inv[e]. All other pairs were already valid.
    
    best_len = 0
    best_start = 1
    
    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Self-inverse elements cannot be in any valid interval
        if e == 1 or e == MOD - 1:
            s = e + 1
            continue
            
        # If inv[e] is already in the window [s, e-1], we must move s past it
        ie = inv[e]
        if ie >= s and ie < e:
            s = ie + 1
            
        # Now [s, e] is valid (no pair x,y in [s,e] has x*y % MOD == 1)
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    if best_len == 0:
        return [2, 1]
    
    # Fast O(N) verification using precomputed inverses
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
    
    if not is_valid(best_start, best_len):
        # Fallback: exhaustive scan (should not trigger if two-pointer is correct)
        for s2 in range(1, N + 1):
            if time.time() > deadline:
                break
            if s2 == 1 or s2 == MOD - 1:
                continue
            for e2 in range(s2, N + 1):
                if e2 == 1 or e2 == MOD - 1:
                    break
                if not is_valid(s2, e2 - s2 + 1):
                    break
                if e2 - s2 + 1 > best_len:
                    best_len = e2 - s2 + 1
                    best_start = s2
    
    return [best_start, best_len]
