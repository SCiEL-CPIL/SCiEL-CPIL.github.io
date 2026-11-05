import time

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill

    # Precompute inverses
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse elements: 1 and 1008 (since 1008 = -1 mod 1009)
    # These cannot be in the interval, so L >= 2 and R <= 1007.

    # Build list of inverse pairs (a, b) where a < b, both in [2, 1007]
    # For each a, we only need the minimum b (most constraining)
    min_b_for_a = [1008] * (MOD + 1)
    for x in range(2, 1008):
        y = inv[x]
        if y > x and 2 <= y <= 1007:
            if y < min_b_for_a[x]:
                min_b_for_a[x] = y

    # Suffix minimum: suffix_min_b[i] = min { b | a >= i, (a,b) in pairs }
    suffix_min_b = [1008] * (MOD + 2)
    for i in range(1007, 1, -1):
        suffix_min_b[i] = min(suffix_min_b[i + 1], min_b_for_a[i])

    best_start = 2
    best_len = 1  # [2,2] is always valid since 2*2=4 != 1

    # For each L, the max valid R is suffix_min_b[L] - 1
    # Length = max_R - L + 1 = suffix_min_b[L] - 1 - L + 1 = suffix_min_b[L] - L
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

    # Final verification pass: confirm the found interval is valid
    # (defensive check, should always pass)
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
