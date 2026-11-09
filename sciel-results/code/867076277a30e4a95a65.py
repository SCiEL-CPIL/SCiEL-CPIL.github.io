<<<<<<< SEARCH
def _local_search(points, max_iter=2000, tol=1e-10):
    """
    Local search: iteratively move the point(s) in the minimum-area triangle
    to increase the minimum area.
    """
    points = points.copy()
    best_area = _min_area(points)
    
    for iteration in range(max_iter):
        min_a, min_idx = _min_area_and_idx(points)
        
        # Get the three points in the minimum triangle
        i, j, k = _idx_i[min_idx], _idx_j[min_idx], _idx_k[min_idx]
        
        improved = False
        # Try moving each of the three points
        for idx in [i, j, k]:
            best_local = points[idx].copy()
            best_local_area = min_a
            
            # Grid search around current position
            for dx in np.linspace(-0.05, 0.05, 11):
                for dy in np.linspace(-0.05, 0.05, 11):
                    trial = points.copy()
                    trial[idx] = [points[idx][0] + dx, points[idx][1] + dy]
                    trial = _clamp_to_triangle(trial)
                    trial_area = _min_area(trial)
                    if trial_area > best_local_area:
                        best_local_area = trial_area
                        best_local = trial[idx].copy()
            
            if best_local_area > min_a + 1e-12:
                points[idx] = best_local
                improved = True
        
        if not improved:
            break
        
        best_area = _min_area(points)
    
    return points
=======
def _local_search(points, time_limit=5.0):
    """
    Fast local search: iteratively move the point(s) in the minimum-area triangle
    to increase the minimum area using 1D scalar minimization.
    """
    points = points.copy()
    start_time = time.time()
    max_iter = 1000
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        min_a, min_idx = _min_area_and_idx(points)
        
        # Get the three points in the minimum triangle
        i, j, k = _idx_i[min_idx], _idx_j[min_idx], _idx_k[min_idx]
        
        improved = False
        
        # Try moving each of the three points
        for idx in [i, j, k]:
            # Define objective for 1D search: minimize -area
            def obj_1d(coord, p_idx=idx, p_x=points[p_idx][0], p_y=points[p_idx][1]):
                trial = points.copy()
                if coord == 0:
                    trial[p_idx] = [p_x + p_y, p_y] # This is a hack, better to use x,y separately
                # Actually, let's just do x and y separately to be safe and simple
                return 0.0 # Placeholder
                
            # Better approach: separate x and y optimizations
            # Optimize x
            def obj_x(x_val, p_idx=idx, p_y=points[p_idx][1]):
                trial = points.copy()
                trial[p_idx] = [x_val, p_y]
                trial = _clamp_to_triangle(trial)
                return -_min_area(trial)
            
            res_x = minimize_scalar(obj_x, bounds=(0, 1), method='bounded', options={'maxiter': 20})
            if res_x.success and -res_x.fun > min_a + 1e-10:
                points[idx][0] = res_x.x
                improved = True
            
            # Optimize y
            def obj_y(y_val, p_idx=idx, p_x=points[p_idx][0]):
                trial = points.copy()
                trial[p_idx] = [p_x, y_val]
                trial = _clamp_to_triangle(trial)
                return -_min_area(trial)
            
            res_y = minimize_scalar(obj_y, bounds=(0, H), method='bounded', options={'maxiter': 20})
            if res_y.success and -res_y.fun > min_a + 1e-10:
                points[idx][1] = res_y.x
                improved = True
        
        if not improved:
            break
        
    return points
>>>>>>> REPLACE

