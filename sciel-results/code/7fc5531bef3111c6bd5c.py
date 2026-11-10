<<<<<<< SEARCH
    # Precompute conflict matrix: conflict[a][b] = set of x in [1,128] that would
    # form a forbidden triple with a and b (i.e., gcd(a,x)=gcd(b,x)=gcd(a,b))
    conflict = [None] * 129
    for a in range(1, 129):
        for b in range(a + 1, 129):
            g_ab = get_gcd(a, b)
            s = set()
            for x in range(1, 129):
                if x == a or x == b:
                    continue
                if get_gcd(a, x) == g_ab and get_gcd(b, x) == g_ab:
                    s.add(x)
            conflict[a] = conflict[a] if conflict[a] is not None else [set() for _ in range(129)]
            conflict[a][b] = s
            conflict[b][a] = s
    # Initialize missing rows
    for a in range(1, 129):
        if conflict[a] is None:
            conflict[a] = [set() for _ in range(129)]
=======
    # Precompute conflict matrix: conflict[a][b] = set of x in [1,128] that would
    # form a forbidden triple with a and b (i.e., gcd(a,x)=gcd(b,x)=gcd(a,b))
    conflict = [[set() for _ in range(129)] for _ in range(129)]
    for a in range(1, 129):
        for b in range(a + 1, 129):
            g_ab = get_gcd(a, b)
            s = set()
            for x in range(1, 129):
                if x == a or x == b:
                    continue
                if get_gcd(a, x) == g_ab and get_gcd(b, x) == g_ab:
                    s.add(x)
            conflict[a][b] = s
            conflict[b][a] = s
>>>>>>> REPLACE
