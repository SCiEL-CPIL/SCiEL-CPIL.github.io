<<<<<<< SEARCH
def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on convex hull facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
    
    facet_angles = []
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        facet_angles.append(angle)
    
    if not facet_angles:
        return np.pi
    
    facet_angles = np.array(facet_angles)
    
    # Soft-max over facet angles
    beta_facet = beta * facet_angles
    max_val = np.max(beta_facet)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_facet - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max
=======
def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on convex hull facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
    
    facet_angles = []
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        facet_angles.append(angle)
    
    if not facet_angles:
        return np.pi
    
    facet_angles = np.array(facet_angles)
    
    # Soft-max over facet angles
    beta_facet = beta * facet_angles
    max_val = np.max(beta_facet)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_facet - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max
>>>>>>> REPLACE
<<<<<<< SEARCH
def _smooth_gradient_optimize(points, time_limit=30, beta=200.0, use_facet=False, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if use_facet:
        grid = None
    elif grid is None:
        grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        if use_facet:
            return _facet_smooth_surrogate(params, beta=beta)
        else:
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)
=======
def _smooth_gradient_optimize(points, time_limit=30, beta=200.0, use_facet=False, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if use_facet:
        grid = None
    elif grid is None:
        grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        if use_facet:
            return _facet_smooth_surrogate(params, beta=beta)
        else:
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)
>>>>>>> REPLACE
<<<<<<< SEARCH
def _spherical_lloyd_iteration(points, max_iter=100, time_limit=30, damping=0.3):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) with adaptive damping.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    current_damping = damping
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        new_points = points.copy()
        
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            normal_sum = np.zeros(3)
            weight_sum = 0.0
            for fi in facets:
                simplex = hull.simplices[fi]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                if np.dot(n, p1) < 0:
                    n = -n
                
                # Weight by facet area (proportional to norm_n)
                weight = norm_n
                normal_sum += weight * n
                weight_sum += weight
            
            if weight_sum > 1e-10:
                target = normal_sum / weight_sum
                target = target / np.linalg.norm(target)
                new_points[pi] = points[pi] + current_damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            # Slowly reduce damping for convergence
            current_damping *= 0.95
        else:
            # Try with smaller damping
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points
=======
def _worst_facet_relaxation(points, max_iter=50, time_limit=15, step_size=0.05):
    """
    Geometric relaxation targeting the worst facet:
    Identify the convex hull facet with largest spherical circumradius,
    then project its three vertices along negative spherical gradients to shrink it.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    current_cost = best_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        # Find the worst facet (largest circumradius angle)
        worst_fi = -1
        worst_angle = 0.0
        for fi, simplex in enumerate(hull.simplices):
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            if np.dot(n, p1) < 0:
                n = -n
            
            h = np.dot(n, p1)
            h = np.clip(h, -1.0, 1.0)
            angle = np.arccos(h)
            if angle > worst_angle:
                worst_angle = angle
                worst_fi = fi
        
        if worst_fi < 0:
            break
        
        simplex = hull.simplices[worst_fi]
        p1_idx, p2_idx, p3_idx = simplex
        
        # Compute gradient directions for each vertex of the worst facet
        # Move each vertex slightly toward the facet center (inward) to shrink the facet
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p3_idx]
        
        # Facet centroid (spherical)
        centroid = p1 + p2 + p3
        centroid = centroid / np.linalg.norm(centroid)
        
        # For each vertex, move toward centroid with small step
        new_points = points.copy()
        for idx in [p1_idx, p2_idx, p3_idx]:
            p = points[idx]
            # Gradient direction: toward centroid
            direction = centroid - p
            direction = direction / np.linalg.norm(direction)
            new_points[idx] = p + step_size * direction
            new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            # Reduce step size and try again
            step_size *= 0.5
            if step_size < 0.001:
                break
    
    return best_points
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Phase 5: Another facet-based pass with high beta
    if _time_left() > 12:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 8), beta=300.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: Final High-Precision Polish with Nelder-Mead
    if _time_left() > 8:
        pts = _nelder_mead_optimize(best_points, time_limit=min(5, _time_left() - 5), maxiter=5000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: One more Nelder-Mead polish if time allows
    if _time_left() > 4:
        pts = _nelder_mead_optimize(best_points, time_limit=min(3, _time_left() - 3), maxiter=3000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
=======
    # Phase 5: Another facet-based pass with high beta
    if _time_left() > 12:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 8), beta=300.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5.5: Worst-facet geometric relaxation
    if _time_left() > 10:
        pts = _worst_facet_relaxation(best_points, max_iter=30, time_limit=min(5, _time_left() - 7), step_size=0.05)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: Final High-Precision Polish with Nelder-Mead
    if _time_left() > 7:
        pts = _nelder_mead_optimize(best_points, time_limit=min(4, _time_left() - 4), maxiter=5000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: One more Nelder-Mead polish if time allows
    if _time_left() > 4:
        pts = _nelder_mead_optimize(best_points, time_limit=min(3, _time_left() - 3), maxiter=3000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
>>>>>>> REPLACE
BET: {mechanism: "Adding a targeted worst-facet geometric relaxation that directly shrinks the largest circumradius facet by moving its vertices toward the facet centroid, complementing the existing Lloyd and gradient-based phases", q: 0.45, delta: [0.0, 0.02]}
