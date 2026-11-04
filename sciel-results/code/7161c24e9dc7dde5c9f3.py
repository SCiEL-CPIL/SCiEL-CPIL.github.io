    n = 11
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_outer_angle = seed_angle
    
    # Initial guess
    init_params = np.zeros(3 * n + 1)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a
    init_params[3*n] = seed_angle

    # Optimization
    try:
        res = minimize(_objective, init_params, args=(n,), method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_side:
            best_inner = []
            for i in range(n):
                best_inner.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            best_outer_angle = res.x[3*n]
            best_side = res.fun
    except Exception:
        pass

    # ... polish ...
