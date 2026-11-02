def construct():
    global START_TIME
    if 'START_TIME' not in dir():
        START_TIME = time.time()
    
    best_result = None
    best_score = np.inf
    
    # Initial guess: honeycomb-like arrangement
    # 12 hexagons in a roughly circular/honeycomb pattern
    # Honeycomb: center + ring of 6 + ring of 6? That's 13. Let's do 1+6+5 or 2 rings.
    # Actually, let's place them in a hexagonal grid pattern.
    
    # Hex grid positions for 12 hexagons (side 1, so center-to-center = sqrt(3) for adjacent)
    d = np.sqrt(3.0)  # center-to-center distance for adjacent hexagons
    
    # Pattern: rows of hexagons
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons (offset)
    # Row 3: 2 hexagons
    # Total: 12
    
    initial_positions = []
    # Use axial coordinates for hex grid
    # Let's just place them in a compact cluster
    # Center at origin, then surrounding
    
    # 1 in center, 6 in first ring, 5 in second ring (partial)
    # First ring: 6 hexagons at distance d, angles 0, 60, 120, 180, 240, 300
    # Second ring: pick 5 from the 12 possible second-ring positions
    
    initial_positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        ang = k * 60.0
        x = d * np.cos(np.radians(ang))
        y = d * np.sin(np.radians(ang))
        initial_positions.append((x, y, 0.0))
    
    # Second ring: 12 positions, pick 5
    # Second ring positions: at distance 2*d or d*sqrt(7) etc.
    # In hex grid, second ring is at various positions
    # Let's just use a simple approach: add 5 more in a ring at larger radius
    for k in range(5):
        ang = k * 72.0 + 30.0  # offset to avoid collision
        r = 2.0 * d
        x = r * np.cos(np.radians(ang))
        y = r * np.sin(np.radians(ang))
        initial_positions.append((x, y, 0.0))
    
    # Convert to flat params
    params0 = np.array([p for pos in initial_positions for p in pos])
    
    # Optimize
    bounds = [(-5, 5), (-5, 5), (-180, 180)] * N
    
    # Use differential evolution for global search, then polish
    elapsed = time.time() - START_TIME
    if elapsed < 60:
        try:
            res = differential_evolution(
                objective, bounds, seed=42, maxiter=50, tol=1e-4,
                popsize=15, mutation=(0.5, 1.5), recombination=0.7
            )
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
    else:
        best_params = params0
    
    elapsed = time.time() - START_TIME
    if elapsed < 100 and 'best_params' in dir():
        # Polish with L-BFGS-B
        try:
            res2 = minimize(
                objective, best_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Final: compute the enclosing hexagon for the best params
    inner = np.array(best_params).reshape(N, 3)
    
    # Find best outer angle
    best_outer = None
    best_side_final = np.inf
    pts = all_vertices(inner)
    
    for outer_ang in np.arange(0, 60, 5.0):
        side, cx, cy = min_enclosing_hexagon(pts, outer_ang)
        if side < best_side_final:
            best_side_final = side
            best_outer = (cx, cy, side, outer_ang)
    
    # Try finer angles around the best
    if best_outer is not None:
        for outer_ang in np.arange(best_outer[3] - 5, best_outer[3] + 5, 0.5):
            side, cx, cy = min_enclosing_hexagon(pts, outer_ang)
            if side < best_side_final:
                best_side_final = side
                best_outer = (cx, cy, side, outer_ang)
    
    cx, cy, side, outer_ang = best_outer
    
    # Return in the required format
    inner_list = [[float(inner[i, 0]), float(inner[i, 1]), float(inner[i, 2])] for i in range(N)]
    return inner_list, (float(cx), float(cy)), float(side), float(outer_ang)
