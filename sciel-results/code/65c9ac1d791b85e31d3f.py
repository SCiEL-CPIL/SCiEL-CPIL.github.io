def construct():
    N = 1008
    P = 1009
    
    # Precompute inverses
    # x * inv[x] % P == 1
    inv = [0] * (P + 1)
    for i in range(1, P):
        # Fermat's little theorem: a^(p-2) = a^-1
        inv[i] = pow(i, P - 2, P)
        
    L = 1
    max_len = 0
    max_start = 1
    
    for R in range(1, N + 1):
        # Check if R is self-inverse (1 or 1008)
        if R == 1 or R == 1008:
            # R cannot be in any valid interval.
            # We must discard R from the window.
            # Since the window is [L, R-1] and we are adding R,
            # and R is invalid on its own, we set L to R+1.
            L = R + 1
            continue
            
        inv_R = inv[R]
        
        # Check if the inverse of R is in the current window [L, R-1]
        # If inv_R >= L, then it is in the window (since inv_R < R is not guaranteed? 
        # Actually, if x*y=1, x and y are distinct unless x=1 or x=p-1.
        # We already handled 1 and p-1.
        # So for other R, inv_R != R.
        # Is inv_R < R? Not necessarily. E.g., 2 * 505 = 1010 = 1 mod 1009.
        # 2 < 505.
        # What if R=505? inv(505)=2.
        # If L=1, window [1, 504]. Is 2 in [1, 504]? Yes.
        # So we need to move L past 2. L becomes 3.
        
        if L <= inv_R < R:
            # inv_R is in the window
            L = inv_R + 1
        elif R < inv_R:
            # inv_R is not in [L, R-1] because it's greater than R.
            # So no conflict with previous elements in the window.
            # However, we must ensure that when we add R, we don't break future constraints?
            # No, the sliding window invariant is that [L, R] is valid.
            # We only need to check if adding R breaks validity with [L, R-1].
            # The only potential conflict is with inv(R).
            # If inv(R) > R, it is not in [L, R-1]. So no conflict.
            pass
        else:
            # inv_R < L, so not in window. No conflict.
            pass
            
        # Now [L, R] is a valid interval
        current_len = R - L + 1
        if current_len > max_len:
            max_len = current_len
            max_start = L
            
    return [max_start, max_len]
