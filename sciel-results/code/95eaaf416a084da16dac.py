import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Self-inverse elements: x^2 % MOD == 1 => x == 1 or x == MOD-1 = 1008
    self_inverse = set([1, 1008])

    # Two-pointer sweep: for each right endpoint R, find the maximal valid window [L, R]
    # Key insight: adding R to window [L, R-1] only creates a new conflict if inv[R] is in [L, R-1]
    # (or if R is self-inverse). This allows O(1) adjustment per step.
    
    best_start = 1
    best_length = 0

    L = 1
    for R in range(1, N + 1):
        if time.time() > deadline:
            break

        # Check if R itself is self-inverse
        if R in self_inverse:
            # Any window containing R is invalid; reset L past R
            L = R + 1
            continue

        # Check if inv[R] is in [L, R-1]
        # If so, we must push L past inv[R] to avoid the conflict pair
        if inv[R] >= L and inv[R] < R:
            L = max(L, inv[R] + 1)

        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L

    # Cross-check via conflict-boundary gap analysis:
    # For each 2-cycle (a, b) with a < b, define a "forbidden span" [a, b].
    # A valid interval must not contain both a and b.
    # Equivalently, for each such pair, the interval must end before b OR start after a.
    # The longest valid interval is the maximum gap between consecutive "constraint boundaries."
    #
    # Alternative view: build a list of all constraint boundaries and find max gap.
    # For each pair (a, b) with a < b and inv[a] == b:
    #   - If interval starts at s <= a, it must end at e < b
    #   - If interval starts at s > a, no constraint from this pair
    # This is exactly what the two-pointer sweep computes.
    #
    # Let's also verify by direct enumeration of all maximal valid intervals:
    # For each start s, find the maximum end e such that [s, e] is valid.
    
    # Verify the two-pointer result with a direct check
    if best_length > 0:
        s_final = best_start
        e_final = s_final + best_length - 1
        valid = True
        for x in range(s_final, e_final + 1):
            if x in self_inverse:
                valid = False
                break
            if inv[x] >= s_final and inv[x] <= e_final:
                valid = False
                break
            if not valid:
                break

        if not valid:
            # Fallback: direct enumeration
            best_start = 1
            best_length = 0
            for s in range(1, N + 1):
                if time.time() > deadline:
                    break
                if s in self_inverse:
                    continue
                # Find max e >= s such that [s, e] is valid
                e_max = s
                for e in range(s + 1, N + 1):
                    if e in self_inverse:
                        break
                    if inv[e] >= s and inv[e] <= e:
                        break
                    e_max = e
                length = e_max - s + 1
                if length > best_length:
                    best_length = length
                    best_start = s

    if best_length == 0:
        # Fallback: find any valid single element
        for i in range(2, N + 1):
            if i not in self_inverse:
                return [i, 1]
        return [2, 1]

    return [best_start, best_length]
