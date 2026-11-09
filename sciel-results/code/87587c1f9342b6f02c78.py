def construct():
    deadline = time.time() + 145.0  # 5s safety margin for 150s limit
    best_points = None
    best_val = 0.0
    
    # Start with a reasonable initial configuration
    rng = random.Random(42)
    
    # Initial guess: place points somewhat evenly
    initial = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.25, H/4],
        [0.75, H/4],
        [0.5, H/2],
        [0.125, H/8],
        [0.875, H/8],
        [0.375, 3*H/8],
        [0.625, 3*H/8],
        [0.5, 3*H/4],
    ])
    initial = project_to_triangle(initial)
    
    # Evaluate initial
    val = -objective(initial.flatten())
    if val > best_val:
        best_val = val
        best_points = initial.copy()
    
    # Try multiple random restarts with local optimization
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        
        # Generate random initial points
        if restarts == 1:
            x0 = initial.flatten()
        else:
            # Random points in triangle
            x0 = np.zeros(N * 2)
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x0[2*i] = u + 0.5 * v
                x0[2*i+1] = H * v
        
        # Local optimization using Nelder-Mead
        try:
            # Reduce maxiter if time is running out
            time_left = deadline - time.time()
            max_iter = 2000 if time_left > 20 else 500
            
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            pts = project_to_triangle(result.x.reshape(N, 2))
            val = -objective(result.x)
            
            if val > best_val:
                best_val = val
                best_points = pts.copy()
                
        except Exception:
            continue

    # Final refinement if time permits
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            pts = project_to_triangle(result.x.reshape(N, 2))
            val = -objective(result.x)
            if val > best_val:
                best_val = val
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = initial
    
    return project_to_triangle(best_points)
