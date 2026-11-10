import time

def construct():
    start_time = time.time()
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (MOD)
    for i in range(1, MOD):
        # Fermat's little theorem: i^(p-2) mod p
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_len = 0
    best_start = 1
    
    # Iterate over all possible start positions
    for s in range(1, N + 1):
        # If s is 1, any interval starting at 1 contains 1, so 1*1=1 mod 1009. Invalid.
        # We can skip s=1? No, just let the loop handle it.
        # Actually, if s=1, e must be >=1. inv[1]=1. 1 is in [1, 0]? No.
        # Check diagonal: 1*1 = 1. Invalid.
        
        current_len = 0
        for e in range(s, N + 1):
            # Check diagonal condition for e
            # e^2 % MOD == 1 ?
            # This is true for e=1 and e=1008.
            if e == 1 or e == 1008:
                # Interval [s, e] contains e which is self-inverse.
                # Since e is in the interval, it's invalid.
                # Any larger e' > e will also contain e, so it will be invalid.
                # So we can break here for this s.
                break
                
            # Check off-diagonal condition
            # Does the interval [s, e] contain any pair x, y such that y = inv[x]?
            # We just added e. So we check if inv[e] is in [s, e-1].
            # If inv[e] is in [s, e-1], then there is a conflict.
            # If there is a conflict in [s, e], there is a conflict in [s, e+1].
            # So we can break.
            
            ie = inv[e]
            if ie >= s and ie < e:
                break
            
            # If we reach here, [s, e] is valid.
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
            # Time check
            if time.time() - start_time > 50:
                return [best_start, best_len]
                
    return [best_start, best_len]
