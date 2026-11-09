def construct():
    MOD = 1009
    inv = [0] * MOD
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse: 1 and 1008, exclude them
    # Search in [2, 1007]
    lo, hi = 2, 1007
    n = hi - lo + 1
    
    # For each L, find max R such that [L,R] is valid
    # Valid: for all x in [L,R], inv[x] < L or inv[x] > R
    # Equivalently: for all x in [L,R], NOT (L <= inv[x] <= R)
    
    best_len = 0
    best_start = lo
    
    for L in range(lo, hi + 1):
        # Max possible R
        max_R = hi
        for x in range(L, hi + 1):
            y = inv[x]
            if L <= y <= max_R:
                # x is in [L, max_R] and y is in [L, max_R] → invalid
                # We need R < y (since y >= L, the constraint is R < y)
                # But only if x <= R. Since x >= L, x is in [L, R] iff x <= R.
                # If x > R, x isn't in the window, so no constraint.
                # So this only constrains if x <= max_R.
                if x <= max_R:
                    max_R = min(max_R, y - 1)
                    if max_R < L:
                        break
        if max_R >= L:
            length = max_R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
    
    return [best_start, best_len]
