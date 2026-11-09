    # Base construction: LUW graph
    q = k
    # For LUW graph 2*q^d, girth is at least 2*d + 1.
    # We need 2*d + 1 >= g => d >= (g - 1) / 2.
    # So d = ceil((g - 1) / 2).
    d = (g + 1) // 2
    if d < 1:
        d = 1
