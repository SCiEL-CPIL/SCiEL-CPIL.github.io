import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # Pairs (a, b) with a, b in [2, 1007] and a*b == 1 mod 1009
    # We only care about distinct pairs for the "blocking" logic, 
    # but self-inverses are handled by range exclusion.
    # Actually, if a==b, a^2=1, a=1 or 1008. Excluded from [2, 1007].
    # So all pairs in [2, 1007] are distinct.
    
    # Group b by L_limit = min(a, b)
    # We want Z(L) = min { b | min(a, b) >= L }
    # This is equivalent to:
    # buckets[L_limit] contains all b such that min(a, b) == L_limit
    # Then Z(L) = min( Z(L+1), min(buckets[L]) )
    
    buckets = [[] for _ in range(MOD + 1)]
    
    for x in range(2, 1008):
        y = inv[x]
        if 2 <= y <= 1007 and y != x:
            # We have a pair (x, y)
            # We need to add y to the bucket for L_limit = min(x, y)
            # Note: The pair is (x, y). The value we are tracking is b.
            # In the formula Z(L) = min { b | ... }, b is the inverse of some a.
            # Here, if we pick a=x, b=y.
            # The condition is min(x, y) >= L.
            # So we add y to bucket[min(x, y)].
            # But wait, the pair is symmetric.
            # If we consider a=y, b=x, we add x to bucket[min(x, y)].
            # Does it matter which one is 'b'?
            # Z(L) is the minimum value of the element that is IN the interval [L, R] 
            # which forces the OTHER element to be OUTSIDE.
            # Actually, the condition is:
            # If we include x, we cannot include y.
            # If we include y, we cannot include x.
            # So if L <= min(x, y), then BOTH x and y are >= L.
            # If we set R < min(x, y), then BOTH are > R? No.
            # If R < min(x, y), then R < x and R < y.
            # So neither x nor y is in [L, R] if L > R? No, L <= R.
            # If L <= min(x, y), then x >= L and y >= L.
            # If we want to include x, we must exclude y. So R < y.
            # If we want to include y, we must exclude x. So R < x.
            # To include ANY number in the interval, we must ensure we don't include its inverse.
            # The "bottleneck" is the smaller of the two?
            # If we set R < min(x, y), then both x and y are > R.
            # So NEITHER is in the interval [L, R].
            # This is safe.
            # Can we set R >= min(x, y)?
            # Suppose min(x, y) = x. So x <= y.
            # If we set R >= x, then x is in [L, R] (since x >= L).
            # Then we MUST have y > R.
            # So R < y.
            # So the constraint is R < y.
            # Note that y >= x >= L.
            # So the constraint is R < y.
            # This implies that for the pair (x, y) with x <= y, the constraint is R < y.
            # This constraint is active only if x >= L (so x is in range).
            # If x < L, then x is not in the interval, so we don't care about y?
            # Wait, if x < L, then x is not in [L, R].
            # But y might be in [L, R].
            # If y is in [L, R], then we must exclude x.
            # But x < L, so x is already outside.
            # So no constraint on R from this pair?
            # Correct.
            # So, for a pair (x, y) with x <= y:
            # The constraint R < y is active IF x >= L.
            # So for L <= x, we have a constraint R < y.
            # So for all L in [2, x], the constraint R < y applies.
            
            # So we should add y to the "active constraints" for all L <= x.
            # This matches the bucket logic:
            # L_limit = x = min(x, y).
            # Add y to bucket[x].
            
            buckets[x].append(y)
            
    # Now compute Z(L) for L from 1007 down to 2
    # Z(L) = min( Z(L+1), min(buckets[L]) )
    # Initialize Z(1008) = 1008 (infinity)
    Z = [1008] * (MOD + 2)
    
    for L in range(1007, 1, -1):
        min_b_in_bucket = 1008
        if buckets[L]:
            min_b_in_bucket = min(buckets[L])
        Z[L] = min(Z[L + 1], min_b_in_bucket)
        
    best_start = 2
    best_len = 1
    
    for L in range(2, 1008):
        # Max R is Z[L] - 1
        max_R = Z[L] - 1
        if max_R >= L:
            length = max_R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
    return [best_start, best_len]
