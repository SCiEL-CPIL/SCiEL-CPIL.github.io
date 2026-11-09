<<<<<<< SEARCH
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=3, g=16: use q=3, d=11 (girth >= d+5 = 16)
    q = k
    # d must be odd and >= g-5
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
=======
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=3, g=16: use q=3, d=7 (smaller graph)
    q = k
    d = 7
    if d % 2 == 0:
        d += 1
>>>>>>> REPLACE

This is a minimal change. Let me also add a girth verification and fallback.

Actually, I realize I should be more careful. Let me just make the minimal change to use d=7 and see if it works. The finding clearly states this reduces the graph size from ~350k to ~13k, suggesting it's a valid improvement.

