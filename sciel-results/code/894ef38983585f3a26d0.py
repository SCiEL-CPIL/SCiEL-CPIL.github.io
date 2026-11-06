def construct():
    best = None
    best_val = float('inf')
    
    # Start from Rudin-Shapiro
    start = _rs_poly()
    c, v = _local_search(start)
    if v < best_val:
        best, best_val = c, v
        
    # Random restarts
    rng = np.random.default_rng(42)
    attempts = 0
    while _time_left() > 10.0 and attempts < 100:
        attempts += 1
        s = [1 if rng.random() < 0.5 else -1 for _ in range(_N)]
        c, v = _local_search(s, max_rounds=20)
        if v < best_val:
            best, best_val = c, v
            # Optional: restart from the best found so far to polish
            # c2, v2 = _local_search(best, max_rounds=50)
            # if v2 < best_val: best, best_val = c2, v2

    if best is None:
        best = start
        
    return [1 if x > 0 else -1 for x in best]
