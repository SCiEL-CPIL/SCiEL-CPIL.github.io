def max_discrepancy(s, limit=None):
    n = len(s)
    max_abs = 0.0
    for step in range(1, n + 1):
        for r in range(step):
            sub = s[r::step]
            if len(sub) == 0:
                continue
            cs = np.cumsum(sub)
            m = np.max(np.abs(cs))
            if m > max_abs:
                max_abs = m
                if limit is not None and max_abs > limit:
                    return max_abs
    return max_abs
