<<<<<<< SEARCH
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
=======
    q = k
    d = (g - 1) // 2
    while 2 * d + 1 < g:
        d += 1
    d = max(3, d)
>>>>>>> REPLACE
