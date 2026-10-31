def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Grid Init + Penalty
    if time.time() < deadline - 10:
        pos, radii = _init_grid()
        p, r = _solve_penalty(pos, radii, deadline)
        if p is not None:
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 2. Random Inits + Penalty
    rng = np.random.RandomState(123)
    for i in range(10):
        if time.time() > deadline - 10:
            break
        pos = rng.uniform(0.1, 0.9, (N, 2))
        radii = rng.uniform(0.05, 0.1, N)
        pos, radii = _post_process(pos, radii)
        
        p, r = _solve_penalty(pos, radii, deadline)
        if p is not None:
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Polish with SLSQP
    if best_pos is not None and time.time() < deadline - 5:
        p, r = _solve_slsqp_vectorized(best_pos, best_radii, deadline)
        if p is not None:
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    if best_pos is None:
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])
