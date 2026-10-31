<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = -float('inf')
    best_points_bary = None
    
    # Generate initializations
    inits = []
    
    # 1. Grid-based init
    try:
        grid_init = _generate_grid_init()
        inits.append(grid_init)
    except Exception:
        pass
        
    # 2. Random inits
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Sample from Dirichlet distribution to get valid barycentric coords
        # alpha=1 gives uniform distribution over the simplex
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2]) # Take u, v
        
    # 3. Perturbed grid init
    try:
        grid_perturbed = _generate_grid_init() + rng.uniform(-0.05, 0.05, size=(N, 2))
        # Clamp to valid range roughly
        grid_perturbed[:, 0] = np.clip(grid_perturbed[:, 0], 0, 1)
        grid_perturbed[:, 1] = np.clip(grid_perturbed[:, 1], 0, 1)
        # Ensure u+v <= 1
        sum_uv = grid_perturbed[:, 0] + grid_perturbed[:, 1]
        mask = sum_uv > 1
        if np.any(mask):
            factor = 1.0 / sum_uv[mask]
            grid_perturbed[mask, 0] *= factor
            grid_perturbed[mask, 1] *= factor
        inits.append(grid_perturbed)
    except Exception:
        pass
        
    # Optimization loop
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            # Flatten init for optimizer
            x0 = init.flatten()
            
            # Bounds for u and v are [0, 1]
            bounds = [(0, 1)] * (2 * N)
            
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Check result
            uv_opt = res.x.reshape(N, 2)
            u_opt = uv_opt[:, 0]
            v_opt = uv_opt[:, 1]
            w_opt = 1 - u_opt - v_opt
            
            # Check validity
            if np.any(u_opt < -1e-5) or np.any(v_opt < -1e-5) or np.any(w_opt < -1e-5):
                continue
                
            # Calculate score
            score = _min_area_bary(uv_opt)
            
            if score > best_score:
                best_score = score
                best_points_bary = uv_opt.copy()
                
        except Exception:
            pass
            
    # If no valid points found, use the grid init
    if best_points_bary is None:
        best_points_bary = _generate_grid_init()
        
    # Convert barycentric to Cartesian
    # Reference triangle: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points_bary[i]
        w = 1 - u - v
        # Clamp small negatives to 0 for numerical stability
        u = max(0, u)
        v = max(0, v)
        w = max(0, w)
        # Normalize if necessary (should be 1)
        s = u + v + w
        if s > 0:
            u /= s
            v /= s
            w /= s
            
        p = u * A + v * B + w * C
        pts.append(p)
        
    return np.array(pts)
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # Reference triangle: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    best_score = -float('inf')
    best_points = None
    
    rng = np.random.RandomState(42)
    
    # Stage 1: Global search using differential evolution
    if time.time() < deadline - 60:
        try:
            # Define bounds for each coordinate
            # x: [0, 1], y: [0, sqrt(3)/2]
            bounds_de = [(0, 1), (0, np.sqrt(3)/2)] * N
            
            def de_objective(params):
                pts = params.reshape(N, 2)
                # Check if all points are inside triangle
                for i in range(N):
                    if not _is_inside_triangle(pts[i], A, B, C):
                        return 100.0  # Large penalty
                min_a = _min_area_cartesian(pts)
                return -min_a
            
            de_result = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            if de_result.success or de_result.fun < 100:
                pts_de = de_result.x.reshape(N, 2)
                pts_de = _clamp_to_simplex_cartesian(pts_de, A, B, C)
                score_de = _min_area_cartesian(pts_de)
                
                if score_de > best_score:
                    best_score = score_de
                    best_points = pts_de.copy()
                    
        except Exception:
            pass
    
    # Stage 2: Local refinement using Nelder-Mead from multiple starting points
    if time.time() < deadline - 10:
        inits = _generate_initializations_cartesian(A, B, C, rng)
        
        # Add the DE result if we have one
        if best_points is not None:
            inits.append(best_points.copy())
        
        # Also add perturbed versions of the best found so far
        for _ in range(5):
            if best_points is not None:
                perturbed = best_points + rng.uniform(-0.03, 0.03, size=(N, 2))
                perturbed = _clamp_to_simplex_cartesian(perturbed, A, B, C)
                inits.append(perturbed)
        
        for init in inits:
            if time.time() > deadline - 5:
                break
            
            try:
                # Nelder-Mead optimization in Cartesian space
                res = minimize(
                    lambda p: _objective_cartesian(p, A, B, C),
                    init.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                pts_opt = res.x.reshape(N, 2)
                pts_opt = _clamp_to_simplex_cartesian(pts_opt, A, B, C)
                
                # Verify all points are inside
                valid = True
                for i in range(N):
                    if not _is_inside_triangle(pts_opt[i], A, B, C):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                score = _min_area_cartesian(pts_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = pts_opt.copy()
                    
            except Exception:
                pass
    
    # If no valid points found, use a fallback
    if best_points is None:
        # Fallback: simple grid
        pts = np.array([
            A, B, C,
            (A + B) / 2.0, (B + C) / 2.0, (C + A) / 2.0,
            (A + B + C) / 3.0,
            A + 0.25 * (B - A) + 0.25 * (C - A),
            B + 0.25 * (A - B) + 0.25 * (C - B),
            C + 0.25 * (A - C) + 0.25 * (B - C),
            A + 0.5 * (B - A) + 0.1 * (C - A)
        ])
        best_points = _clamp_to_simplex_cartesian(pts, A, B, C)
    
    return best_points
>>>>>>> REPLACE
