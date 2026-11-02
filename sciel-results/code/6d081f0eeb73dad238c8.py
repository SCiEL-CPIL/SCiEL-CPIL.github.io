<<<<<<< SEARCH
    # Initial guess
    # Standard hexagonal packing:
    # Inner ring at distance 2.0 (touching center)
    # Outer ring at distance ~3.46 (touching inner ring)
    # Angles: 0 for inner, 30 for outer (to fit in gaps)
    # Outer rotation: 30 (flat top) or 0 (pointy top)
    
    initial_guess = np.array([2.0, 0.0, 3.5, 30.0, 30.0])
    
    # Bounds for Nelder-Mead (not strictly enforced by NM, but we can clip)
    # r1, r2 > 0
    # Angles 0-360
    
    result = None
    try:
        result = minimize(
            objective, 
            initial_guess, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
        )
    except Exception as e:
        pass

    if result is None or not result.success:
        # Fallback to a known valid configuration if optimization fails
        # Use the initial guess if it's valid
        best_params = initial_guess
    else:
        best_params = result.x
=======
    # Honeycomb initial guess:
    # Inner ring at distance sqrt(3) ≈ 1.732 (touching center hexagon)
    # Outer ring at distance 2*sqrt(3) ≈ 3.464 (touching inner ring)
    # Angles: 0 for inner ring, 0 for outer ring (all aligned)
    # Outer rotation: 0 (pointy top)
    
    sqrt3 = math.sqrt(3)
    initial_guess = np.array([sqrt3, 0.0, 2*sqrt3, 0.0, 0.0])
    
    # First pass: coarse Nelder-Mead
    result = None
    try:
        result = minimize(
            objective, 
            initial_guess, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'initial_simplex': 0.5}
        )
    except Exception as e:
        pass

    if result is not None and result.fun < 100:
        best_params = result.x
    else:
        best_params = initial_guess

    # Second pass: fine polish from the first result
    if time.time() < deadline - 10:
        try:
            result2 = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'initial_simplex': 0.01}
            )
            if result2 is not None and result2.fun < objective(best_params):
                best_params = result2.x
        except Exception as e:
            pass
>>>>>>> REPLACE

