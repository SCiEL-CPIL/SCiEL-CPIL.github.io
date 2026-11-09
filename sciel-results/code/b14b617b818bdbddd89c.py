import time

def construct():
    MOD = 1009
    N = 1008
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin well under the 90s kill

    # Modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverses (x^2 = 1 mod 1009): x = 1 and x = 1008.
    # These cannot be present because x*x % 1009 == 1.
    forbidden = {1, 1008}

    # For each start s, the maximal right end r(s) such that [s, r] is valid.
    # Constraint from element x in the window:
    #   if inv[x] > x: window must end at most at inv[x]-1
    #   if inv[x] < x: window must start at most at inv[x]+1 (handled by s)
    # For a fixed s, r(s) = min over x in [s, r] of (inv[x]-1) where inv[x] > x.
    # r(s) is nondecreasing in s, so two-pointer works.

    best_start = 2
    best_len = 0

    # Precompute upper bound from inv[x] for x where inv[x] > x
    # bound[x] = inv[x] - 1 if inv[x] > x else large
    LARGE = N + 1
    bound = [LARGE] * (N + 1)
    for x in range(1, N + 1):
        ix = inv[x]
        if ix > x:
            bound[x] = ix - 1

    left = 1
    for right in range(1, N + 1):
        # The new element 'right' imposes: window must end at <= bound[right]
        # i.e., left must be at least... no: bound is an upper bound on right.
        # If right > bound[right], then we must move left past 'right'?? No.
        # Correct: for element x=right in window, we need window_end <= bound[right].
        # Since window_end = right, we need right <= bound[right], else invalid.
        # bound[right] < right means inv[right] < right+1, i.e., inv[right] <= right.
        # If inv[right] < right, the conflict is with a smaller element, handled by left.
        # Let's use the standard formulation:
        #   We need for all x in [left, right]: if inv[x] in [left, right] and inv[x] != x, invalid.
        # Equivalent: for all x in [left, right]: inv[x] not in [left, right].
        #   inv[x] >= right+1  OR  inv[x] <= left-1.
        # For fixed left, as right grows, the binding constraint is the first x with inv[x] in (left, right].

        # Two-pointer: maintain left so that window [left, right] is valid.
        # When we add 'right', if inv[right] >= left and inv[right] <= right and inv[right] != right,
        #   we have a conflict. If inv[right] < right, conflict with inv[right] (smaller), push left.
        #   If inv[right] > right, it's not in window yet, no conflict now.
        #   If inv[right] == right, self-inverse, but only 1 and 1008, handled by forbidden.

        # Check forbidden
        if right in forbidden:
            # 'right' cannot be in any valid window, so it acts as a hard barrier.
            # Actually it means any window containing 'right' is invalid.
            # So we must move left past 'right' (i.e., left = right+1) and skip.
            left = right + 1
            continue

        # Check conflict from 'right' with elements already in [left, right-1]
        ix = inv[right]
        if left <= ix < right:
            # ix is in the window and ix != right, conflict. Push left past ix.
            left = ix + 1
        # If ix > right, no conflict now (ix not yet in window).
        # If ix < left, no conflict (ix already excluded).

        if left < 1:
            left = 1
        if left > right:
            continue

        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left

        if time.time() > deadline:
            break

    if best_len <= 0:
        # Fallback: a single element that is not self-inverse and whose inverse is not itself.
        # 2 is valid: 2*2=4 !=1, and inv[2]=505 not in {2}.
        best_start = 2
        best_len = 1

    # Validate the best candidate; if invalid, shrink or search locally.
    def is_valid(s, ln):
        if s < 1 or s + ln - 1 > N:
            return False
        for x in range(s, s + ln):
            if x in forbidden:
                return False
            ix = inv[x]
            if s <= ix <= s + ln - 1 and ix != x:
                return False
        return True

    if not is_valid(best_start, best_len):
        # Fallback exhaustive small search
        found = False
        for ln in range(1, best_len + 1):
            for s in range(1, N - ln + 2):
                if is_valid(s, ln):
                    best_start = s
                    best_len = ln
                    found = True
                    break
            if found:
                break
        if not found:
            best_start = 2
            best_len = 1

    return [best_start, best_len]
