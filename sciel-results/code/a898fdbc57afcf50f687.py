<<<<<<< SEARCH
def optimize_topology(centers, angles, max_time=60.0):
    """Optimize positions and angles to minimize outer side."""
    t0 = time.time()
    
    # Initial parameters
    params = []
    for i in range(N):
        params.append(centers[i][0])
        params.append(centers[i][1])
        params.append(math.radians(angles[i]))
    
    bounds = [(-5, 5)] * (2 * N) + [(-0.52, 0.52)] * N
    
    # Use differential evolution for global search
    def obj_func(p, oa):
        return objective(p, oa)
    
    try:
        result = differential_evolution(
            lambda p: obj_func(p, 0.0),
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False
        )
        best_params = result.x
        best_val = result.fun
    except Exception:
        best_params = params
        best_val = objective(params, 0.0)
    
    # Local refinement with Nelder-Mead
    if time.time() - t0 < max_time - 5:
        try:
            local_res = minimize(
                lambda p: objective(p, 0.0),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if local_res.fun < best_val:
                best_params = local_res.x
                best_val = local_res.fun
        except Exception:
            pass
    
    # Extract results
    centers_out = [(best_params[2 * i], best_params[2 * i + 1]) for i in range(N)]
    angles_out = [math.degrees(best_params[2 * i + 2]) for i in range(N)]
    outer_side = compute_outer_side(centers_out, angles_out, 0.0)
    
    return outer_side, centers_out, angles_out
=======
def optimize_topology(centers, angles, max_time=60.0):
    """Optimize positions and angles to minimize outer side."""
    t0 = time.time()
    
    # Initial parameters
    params = []
    for i in range(N):
        params.append(centers[i][0])
        params.append(centers[i][1])
        params.append(math.radians(angles[i]))
    
    bounds = [(-5, 5)] * (2 * N) + [(-0.52, 0.52)] * N
    
    best_params = params[:]
    best_val = objective(params, 0.0)
    
    # Use differential evolution for global search - REDUCED PARAMETERS FOR SPEED
    # We check time before and after
    if time.time() - t0 < max_time * 0.5:
        try:
            # Reduced popsize and maxiter to ensure completion
            result = differential_evolution(
                lambda p: objective(p, 0.0),
                bounds,
                maxiter=10,
                popsize=5,
                tol=1e-3,
                seed=42,
                polish=False
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() - t0 < max_time - 2:
        try:
            # Reduced maxiter for local search
            local_res = minimize(
                lambda p: objective(p, 0.0),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if local_res.fun < best_val:
                best_params = local_res.x
                best_val = local_res.fun
        except Exception:
            pass
    
    # Extract results
    centers_out = [(best_params[2 * i], best_params[2 * i + 1]) for i in range(N)]
    angles_out = [math.degrees(best_params[2 * i + 2]) for i in range(N)]
    outer_side = compute_outer_side(centers_out, angles_out, 0.0)
    
    return outer_side, centers_out, angles_out
>>>>>>> REPLACE

