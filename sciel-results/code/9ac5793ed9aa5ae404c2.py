<<<<<<< SEARCH
    q = k
    # Choose d so that the girth is at least d + 5 (for odd d) or d + 4 (even).
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
=======
    q = k
    # The girth of the LUW construction with parameter d is 2d + 2.
    # We require girth >= g, so 2d + 2 >= g => d >= (g - 2) / 2.
    d = max(3, (g - 2 + 1) // 2)
>>>>>>> REPLACE
