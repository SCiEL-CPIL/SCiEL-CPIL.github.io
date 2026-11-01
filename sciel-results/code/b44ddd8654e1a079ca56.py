<<<<<<< SEARCH
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
=======
    q = k
    # Minimal d for LUW construction to achieve girth g is (g-1)//2
    # LUW graphs have girth at least 2*d + 1
    d = (g - 1) // 2
    if d < 1:
        d = 1
>>>>>>> REPLACE
