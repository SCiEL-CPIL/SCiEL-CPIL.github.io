    # Seed logic for initial valid set
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    seed_points = [(x, 0) for x in xs]
    
    # Use seed points as initial state if random init is bad
    # Or just use seed points as the fallback best
    best_points = seed_points[:]
    best_size = len(seed_points)
    best_C = 0
