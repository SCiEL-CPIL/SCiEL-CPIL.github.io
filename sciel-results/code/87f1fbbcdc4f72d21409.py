<<<<<<< SEARCH
    # Phase 2: Polish with very sharp soft-min
    points = best_points.copy()
    alpha = 0.005
    cooling = 0.9995
    k_soft = 200.0  # Very sharp for polishing
    
    while time.time() - start_time < time_limit - 5.0:  # Leave 5s for final check
=======
    # Phase 2: Polish with very sharp soft-min
    points = best_points.copy()
    alpha = 0.005
    cooling = 0.9995
    k_soft = 200.0  # Very sharp for polishing
    
    while time.time() - start_time < time_limit - 5.0:  # Leave 5s for final check
        # Occasional random restarts in Phase 2 to escape local minima
        if np.random.rand() < 0.01 and (time.time() - start_time) < time_limit - 10.0:
            points = best_points.copy()
            points += np.random.normal(0, 0.002, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
>>>>>>> REPLACE
