<<<<<<< SEARCH
    init_positions, init_angles = best_valid_config
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    # --- PHASE 2: SLSQP REFINEMENT WITH HARD CONSTRAINTS ---
    # Now that we have a valid configuration, we use SLSQP to minimize the outer side length.
    # SLSQP handles inequality constraints directly, which is superior to penalty methods
    # for maintaining strict non-overlap while pushing boundaries.
    
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        # Use a fixed outer angle (0) for the optimization to keep it smooth.
        # We will optimize the outer angle separately in the final step or by including it.
        # For now, let's include outer angle in the parameters? 
        # No, let's keep it simple: optimize inner positions/angles for outer angle 0.
        # Then in Phase 3, we can try to optimize outer angle.
        return calc_min_outer_side(positions, angles, 0.0)
    
    # Define constraints for SLSQP
    # For each pair (i, j), we require gap >= 0.
    # gap = min_separation.
    # We define constraint as gap(i, j) >= 0.
    # SLSQP expects func(x) >= 0 for inequality constraints.
    
    def make_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            # We need a smooth approximation of the gap for gradients?
            # SLSQP uses finite differences or analytical gradients.
            # sat_gap is non-smooth. 
            # However, near the boundary of contact, the gap is determined by the minimum projection.
            # To make it robust, we return the gap. If it's deep overlap, it's negative.
            # The constraint is gap >= 0.
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))

    best_score = best_valid_score
    best_result = x0.copy()

    # Run SLSQP
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 400, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.success or res.fun < best_score:
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:]
            
            # Verify validity strictly
            if is_valid(pos, ang, tol=1e-4):
                # Use the best outer angle for this configuration
                score, _ = find_best_outer_angle(pos, ang, coarse=True)
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
            else:
                # If SLSQP violated constraints slightly, we can try to repair or discard
                # For now, discard if invalid.
                pass
    except Exception as e:
        print(f"SLSQP error: {e}")
=======
    init_positions, init_angles = best_valid_config
    
    # Find initial best outer angle
    init_outer_angle = 0.0
    _, init_outer_angle = find_best_outer_angle(init_positions, init_angles, coarse=True)
    
    # Build initial parameter vector with outer angle
    x0 = np.concatenate([init_positions.flatten(), init_angles, [init_outer_angle]])
    
    # --- PHASE 2: SLSQP REFINEMENT WITH HARD CONSTRAINTS ---
    # Now that we have a valid configuration, we use SLSQP to minimize the outer side length.
    # SLSQP handles inequality constraints directly, which is superior to penalty methods
    # for maintaining strict non-overlap while pushing boundaries.
    
    # Define constraints for SLSQP
    # For each pair (i, j), we require gap >= 0.
    # gap = min_separation.
    # We define constraint as gap(i, j) >= 0.
    # SLSQP expects func(x) >= 0 for inequality constraints.
    
    def make_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))
    bounds.append((0, 60))  # outer angle
    
    best_score = best_valid_score
    best_result = x0.copy()
    
    # Run SLSQP with multiple restarts from perturbed initial points
    np.random.seed(42)
    num_restarts = 5
    max_iter_per_restart = 300
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial point
            perturbation_scale = 0.05 * (restart + 1)
            x_start = x0 + np.random.randn(len(x0)) * perturbation_scale
            # Clamp angles to valid range
            x_start[2*N:3*N] = np.mod(x_start[2*N:3*N], 360)
            x_start[3*N] = np.mod(x_start[3*N], 60)
            # Clamp positions
            x_start[:2*N] = np.clip(x_start[:2*N], -10, 10)
        
        try:
            res = minimize(
                objective, 
                x_start, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter_per_restart, 'ftol': 1e-10, 'disp': False}
            )
            
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:3*N]
            outer_ang = res.x[3*N]
            
            # Verify validity strictly
            if is_valid(pos, ang, tol=1e-4):
                # Use the best outer angle for this configuration
                score, _ = find_best_outer_angle(pos, ang, coarse=True)
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
        except Exception as e:
            pass
>>>>>>> REPLACE

