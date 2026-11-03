<<<<<<< SEARCH
    # Final high-precision polish: refine the outer hexagon parameters
    # Use a more precise local optimization around the current best angle
    if time.time() < deadline - 5.0:  # Leave 5s safety margin
        # Try a few nearby angles with higher precision
        for delta_angle in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            test_angle = best_angle + delta_angle
            if test_angle < 0 or test_angle > 30:
                continue
            side_test, ox_test, oy_test = _min_outer_side_analytical(inner_hexes, math.radians(test_angle))
            if side_test < best_side:
                best_side = side_test
                best_angle = test_angle
                best_center = (ox_test, oy_test)
        
        # One more ternary search refinement in a narrow window
        if time.time() < deadline - 3.0:
            lo = max(0.0, best_angle - 3.0)
            hi = min(30.0, best_angle + 3.0)
            gr = (math.sqrt(5) - 1) / 2
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
            fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
            
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - gr * (hi - lo)
                    fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + gr * (hi - lo)
                    fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
            
            best_angle = (lo + hi) / 2.0
            best_side, ox, oy = _min_outer_side_analytical(inner_hexes, math.radians(best_angle))
            best_center = (ox, oy)
=======
    # Final high-precision polish: refine the outer hexagon parameters
    # Use a more precise local optimization around the current best angle
    if time.time() < deadline - 5.0:  # Leave 5s safety margin
        # First, do a coarse grid search to find the best region
        grid_angles = [best_angle + d for d in range(-15, 16) if 0 <= best_angle + d <= 30]
        for test_angle in grid_angles:
            if time.time() > deadline - 5.0:
                break
            side_test, ox_test, oy_test = _min_outer_side_analytical(inner_hexes, math.radians(test_angle))
            if side_test < best_side:
                best_side = side_test
                best_angle = test_angle
                best_center = (ox_test, oy_test)
        
        # Second, do a fine ternary search in a narrow window around the best angle
        if time.time() < deadline - 3.0:
            lo = max(0.0, best_angle - 2.0)
            hi = min(30.0, best_angle + 2.0)
            gr = (math.sqrt(5) - 1) / 2
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
            fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
            
            # More iterations for higher precision
            for _ in range(40):
                if time.time() > deadline - 2.0:
                    break
                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - gr * (hi - lo)
                    fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + gr * (hi - lo)
                    fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
            
            best_angle = (lo + hi) / 2.0
            best_side, ox, oy = _min_outer_side_analytical(inner_hexes, math.radians(best_angle))
            best_center = (ox, oy)
        
        # Third, do an even finer local search with small perturbations
        if time.time() < deadline - 1.0:
            for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                test_angle = best_angle + delta
                if 0 <= test_angle <= 30:
                    side_test, ox_test, oy_test = _min_outer_side_analytical(inner_hexes, math.radians(test_angle))
                    if side_test < best_side - 1e-12:
                        best_side = side_test
                        best_angle = test_angle
                        best_center = (ox_test, oy_test)
>>>>>>> REPLACE

