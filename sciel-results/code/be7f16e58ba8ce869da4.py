    for d in [8, 9, 10, 11]:
        ...
        n, edges = _luw_construct_with_d(k, g, d)
        if 510 <= n <= 50000:
            if _verify_girth_fast(edges, n, g, time_limit=10.0):
                return n, edges
    