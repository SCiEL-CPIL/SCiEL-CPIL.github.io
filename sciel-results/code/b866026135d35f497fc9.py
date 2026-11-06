def dfs(rem, min_d, terms_left, M, path):
    if rem == 0:
        return True
    if terms_left == 0 or time exceeded:
        return False
    # Bounds
    max_possible = sum of largest terms_left unit fractions in [min_d, M]
    if rem > max_possible: return False
    if rem < 1/M: return False
    # Target a
    a_target = terms_left / rem  # as a float
    # Generate candidate a values near a_target, in [lo, M]
    lo = max(min_d, ceil(1/rem))
    hi = M
    # Order: try a near a_target first
    candidates = list(range(lo, hi+1))
    # Sort by distance to a_target
    candidates.sort(key=lambda a: abs(a - a_target))
    for a in candidates:
        new_rem = rem - 1/a
        if new_rem < 0: continue
        path.append(a)
        if dfs(new_rem, a+1, terms_left-1, M, path):
            return True
        path.pop()
    return False
