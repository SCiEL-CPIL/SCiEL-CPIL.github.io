    # Final polish with Powell's method on the full-resolution objective
    if time.time() < deadline - safety_margin:
        try:
            # Use a small initial step size for Powell
            # Limit iterations to prevent timeout
            result2 = minimize(
                _union_area_full,
                best_x,
                method='Powell',
                options={
                    'maxiter': 10,
                    'ftol': 1e-12,
                    'xtol': 1e-12,
                }
            )
            
            if np.all(np.isfinite(result2.x)):
                area_candidate = _union_area_full(result2.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result2.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
