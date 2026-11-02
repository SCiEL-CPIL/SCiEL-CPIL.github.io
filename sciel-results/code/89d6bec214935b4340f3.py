<<<<<<< SEARCH
    q = k
    # Choose d to ensure girth >= g
    if (g - 5) % 2 == 1 and g - 5 >= 3:
        d = g - 5  # odd, girth >= d + 5 = g
    else:
        d = g - 4  # even, girth >= d + 4 = g
    d = max(3, d)
=======
    # Use q=2 to keep n small while maintaining high girth
    # CD(d, 2) has n = 2 * 2^d = 2^(d+1) vertices
    # Girth >= d + 4 (conservative). We need girth >= g.
    # For g=16, d=13 gives girth >= 17. n = 2^14 = 16384.
    q = 2
    d = g - 4
    if d < 3:
        d = 3
    # Ensure n is not too large (limit 50000)
    # n = 2^(d+1). 2^15 = 32768, 2^16 = 65536.
    # So d+1 <= 15 => d <= 14.
    if d > 14:
        d = 14
>>>>>>> REPLACE
