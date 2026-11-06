import time

def construct():
    MOD = 1009
    N = MOD - 1  # 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # For each position e (right endpoint), we need to find the largest s
    # such that [s, e] contains no pair (x, y) with x*y % MOD == 1.
    #
    # A pair (x, y) with x*y % MOD == 1 means y = inv[x] (and x = inv[y]).
    # So for interval [s, e] to be valid, for every x in [s, e], inv[x] must NOT be in [s, e].
    #
    # This is equivalent to: for every x in [s, e], either inv[x] < s or inv[x] > e.
    #
    # Using two-pointer: maintain s such that [s, e] is valid.
    # When we add e, if inv[e] is in [s, e-1], we must move s past inv[e].
    # Also, if e is self-inverse (e == inv[e]), i.e., e*e % MOD == 1, then e cannot be in any valid interval.
    # Self-inverse elements: e^2 ≡ 1 (mod 1009) => e ≡ 1 or e ≡ -1 ≡ 1008 (mod 1009).
    # So e=1 and e=1008 are self-inverse and must be excluded.

    best_len = 0
    best_start = 1

    s = 1
    for e in range(1, N + 1):
        # If e is self-inverse, it cannot be part of any valid interval
        if e * e % MOD == 1:
            s = e + 1
            continue

        # If inv[e] is in the current window [s, e-1], we must move s past it
        ie = inv[e]
        if ie >= s and ie <= e:  # ie == e would mean self-inverse, already handled
            s = ie + 1

        # Now [s, e] should be valid (no pair x, inv[x] both in [s, e])
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # The answer should be the maximum length found
    return [best_start, best_len]
