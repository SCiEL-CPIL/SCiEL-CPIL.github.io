<<<<<<< SEARCH
    q = k
    # Choose d so that the girth is at least d + 5 (for odd d) or d + 4 (even).
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
=======
    q = k
    # Choose d such that the girth of the LUW graph is at least g.
    # The girth of LUW(q, d) is typically 2d + 1.
    # So we need 2d + 1 >= g => d >= (g - 1) / 2.
    d = (g - 1) // 2
    if d < 2:
        d = 2
>>>>>>> REPLACE
