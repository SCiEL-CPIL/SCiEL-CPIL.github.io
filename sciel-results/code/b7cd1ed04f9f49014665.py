import time

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = 70  # safety margin before 90s kill

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse elements: 1 and 1008 (since 1008 = -1 mod 1009)
    # These cannot be in the interval, so L >= 2 and R <= 1007.

    # For each position x in [2, 1007], determine the constraint it imposes:
    # If x is in the interval, then inv[x] must NOT be in the interval.
    # For a sliding window [L, R], we need that for all x in [L,R], inv[x] is outside [L,R].
    # This means: for all x in [L,R], either inv[x] < L or inv[x] > R.
    # Equivalently: min_{x in [L,R]} inv[x] < L OR max_{x in [L,R]} inv[x] > R
    # But we need BOTH conditions to hold simultaneously for all x, which is equivalent to:
    # For all x in [L,R]: inv[x] < L OR inv[x] > R
    # This is equivalent to: no x in [L,R] has L <= inv[x] <= R.
    
    # Two-pointer sliding window approach:
    # For a fixed L, extend R as far as possible.
    # When we add a new element R to the window, we check if inv[R] is in [L, R].
    # If inv[R] >= L, then we have a conflict (since inv[R] <= R is needed too, but actually
    # we need inv[R] < L or inv[R] > R; since inv[R] <= R always when inv[R] >= L... 
    # wait, inv[R] can be greater than R).
    
    # Let me think more carefully. For window [L, R], the constraint is:
    # For every x in [L, R], inv[x] must be < L or > R.
    # 
    # Using two pointers: start with L=2, R=1 (empty). Extend R while valid.
    # When we add R to the window [L, R-1], the new constraint is: inv[R] < L or inv[R] > R.
    # If inv[R] >= L and inv[R] <= R, then we have a conflict.
    # But also, when we increment L, we remove L from the window, which might resolve conflicts.
    
    # Better approach: for each L, find the maximum R such that [L, R] is valid.
    # As L increases, R can only stay the same or increase (monotonicity).
    # Actually, this isn't quite right either. Let me use a different approach.
    
    # For each potential right endpoint R, what is the minimum L such that [L, R] is valid?
    # Or: for each L, what is the maximum R?
    
    # Let's use the following: for each x in [2, 1007], define:
    #   left_constraint[x] = x if inv[x] > x else (some value that makes the pair invalid)
    # Actually, let me think of it as: for a pair (x, y) where y = inv[x] and x != y,
    # we cannot have both x and y in the window. So if x < y, then we can't have a window
    # that contains both x and y.
    
    # For each x in [2, 1007] where inv[x] != x (i.e., x != 1 and x != 1008, which is already excluded):
    # Let a = min(x, inv[x]), b = max(x, inv[x]).
    # We cannot have a window [L, R] with L <= a and b <= R.
    # This means: for any valid window [L, R], it's NOT the case that L <= a AND b <= R.
    # Equivalently: L > a OR R < b.
    
    # So for a given L, the maximum R is: min over all pairs (a, b) with a >= L of (b - 1),
    # but also we need R < b for all pairs with a < L... no wait.
    
    # For a given L, a pair (a, b) with a < L doesn't constrain R (since L > a is already satisfied).
    # A pair (a, b) with a >= L constrains R < b, i.e., R <= b - 1.
    # So max R for given L = min { b - 1 : (a, b) is a pair with a >= L }
    
    # This is the same as the suffix minimum approach in the original code!
    # Let me verify this is correct and just optimize the implementation.
    
    # Build list of inverse pairs (a, b) where a < b, both in [2, 1007]
    pairs = []
    for x in range(2, 1008):
        y = inv[x]
        if y > x and 2 <= y <= 1007:
            pairs.append((x, y))
    
    # For each L, max_R = min { b - 1 : (a, b) in pairs with a >= L }
    # If no such pair exists, max_R = 1007.
    
    # Suffix minimum of b values
    # suffix_min_b[i] = min { b : (a, b) in pairs with a >= i }
    suffix_min_b = [1008] * (MOD + 2)
    for a, b in pairs:
        if b < suffix_min_b[a]:
            suffix_min_b[a] = b
    
    for i in range(1007, 1, -1):
        if suffix_min_b[i + 1] < suffix_min_b[i]:
            suffix_min_b[i] = suffix_min_b[i + 1]
    
    best_start = 2
    best_len = 1  # [2,2] is always valid since 2*2=4 != 1
    
    for L in range(2, 1008):
        if time.time() - start_time > deadline:
            break
        min_b = suffix_min_b[L]
        max_R = min_b - 1
        if max_R >= L:
            length = max_R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
    
    # Final verification pass
    if time.time() - start_time < deadline - 5:
        R = best_start + best_len - 1
        valid = True
        for x in range(best_start, R + 1):
            ix = inv[x]
            if best_start <= ix <= R:
                valid = False
                break
        if not valid:
            # Fallback: scan forward from best_start to find valid R
            R_max = 1007
            for x in range(best_start, 1008):
                if x > R_max:
                    break
                ix = inv[x]
                if ix >= best_start:
                    R_max = min(R_max, ix - 1)
                    if R_max < best_start:
                        break
            if R_max >= best_start:
                best_len = R_max - best_start + 1
            else:
                best_len = 1
    
    return [best_start, best_len]
