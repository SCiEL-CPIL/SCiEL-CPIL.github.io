def construct():
    global START_TIME, TIME_LIMIT

    inner = []
    # Central
    inner.append([0.0, 0.0, 0.0])
    # Ring 1: 6 hexagons at distance 2.0
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = 2.0 * np.cos(ang)
        y = 2.0 * np.sin(ang)
        inner.append([x, y, 0.0])
    # Ring 2: 4 hexagons at distance 3.5, angles 30, 150, 210, 330
    for ang_deg in [30, 150, 210, 330]:
        ang = np.deg2rad(ang_deg)
        x = 3.5 * np.cos(ang)
        y = 3.5 * np.sin(ang)
        inner.append([x, y, 0.0])

    n = len(inner)
    init_params = []
    for h in inner:
        init_params.extend(h)
    # Outer center and angle
    init_params.extend([0.0, 0.0, 0.0])
    init_params = np.array(init_params)

    best_params = init_params.copy()
    best_val = objective(init_params)

    # Optimization
    if time.time() - START_TIME < TIME_LIMIT - 20.0:
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    if time.time() - START_TIME < TIME_LIMIT - 10.0:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass

    inner_out = []
    for i in range(n):
        inner_out.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    cx = best_params[3*n]
    cy = best_params[3*n+1]
    angle_outer = best_params[3*n+2]
    side = best_val

    return inner_out, (cx, cy), side, angle_outer
