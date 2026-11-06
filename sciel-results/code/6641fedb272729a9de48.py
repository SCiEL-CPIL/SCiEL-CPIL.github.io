def construct():
    n = 1008
    p = 1009

    # Precompute modular inverses mod 1009 for x in [1, 1008]
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)

    # For each x, define lo[x] = min(x, inv[x]) and hi[x] = max(x, inv[x]).
    # An interval [s, e] is valid iff for all x in [1, n], NOT (s <= lo[x] AND hi[x] <= e).
    # For fixed s, the maximum valid e is:
    #   max_e = min over all x where lo[x] >= s of (hi[x] - 1)
    # If no such x exists, max_e = n.

    lo = [0] * (n + 1)
    hi = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = inv[x]
        lo[x] = min(x, ix)
        hi[x] = max(x, ix)

    # min_hi_for_lo[v] = min of hi[x] where lo[x] == v
    min_hi_for_lo = [n + 1] * (n + 2)
    for x in range(1, n + 1):
        if hi[x] < min_hi_for_lo[lo[x]]:
            min_hi_for_lo[lo[x]] = hi[x]

    # Build suffix min: min_hi_at[s] = min(min_hi_for_lo[s], min_hi_for_lo[s+1], ..., min_hi_for_lo[n+1])
    min_hi_at = [n + 1] * (n + 3)
    running_min = n + 1
    for s in range(n + 1, 0, -1):
        if min_hi_for_lo[s] < running_min:
            running_min = min_hi_for_lo[s]
        min_hi_at[s] = running_min

    best_start = 2
    best_length = 1

    for s in range(1, n + 1):
        blocking = min_hi_at[s]
        if blocking > n:
            max_e = n
        else:
            max_e = blocking - 1

        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    return [best_start, best_length]
