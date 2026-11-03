    all_dist_2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            d = max(abs(q), abs(r), abs(-q-r))
            if d == 2:
                all_dist_2.append((q,r))
