def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several initializations
    # 1. Regular polygon with small jitter
    # 2. Regular polygon with larger jitter
    # 3. Random points (normalized)
    
    inits = []
    
    # Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Jitter 1: small
    jitter1 = rng.normal(0, 0.01, base.shape)
    inits.append(base + jitter1)
    
    # Jitter 2: medium
    jitter2 = rng.normal(0, 0.05, base.shape)
    inits.append(base + jitter2)
    
    # Jitter 3: larger
    jitter3 = rng.normal(0, 0.1, base.shape)
    inits.append(base + jitter3)
    
    # Random
    rand_pts = rng.random((N, 2))
    # Normalize to unit area roughly?
    # Just use as is, optimizer will handle scale?
    # Actually, random points in [0,1]x[0,1] have small hull area.
    # It's better to normalize initial points to unit hull area to start in a good scale.
    
    def normalize_unit_area(pts):
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-9:
                return pts / np.sqrt(area)
        except:
            pass
        return pts
        
    inits = [normalize_unit_area(p) for p in inits]
    
    for init in inits:
        if time.time() > deadline - 10: # Leave 10s margin
            break
            
        current_points = init.copy()
        betas = [5, 20, 100, 500, 2000]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            def objective(x):
                points = x.reshape(-1, 2)
                try:
                    hull = ConvexHull(points)
                    hull_area = hull.volume
                    if hull_area < 1e-9:
                        return 1e10
                except Exception:
                    return 1e10
                    
                p1 = points[triangle_indices[:, 0]]
                p2 = points[triangle_indices[:, 1]]
                p3 = points[triangle_indices[:, 2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                areas = 0.5 * np.abs(dets)
                
                z = -beta * areas
                z_max = np.max(z)
                log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
                soft_min = -log_sum_exp / beta
                
                score = soft_min / hull_area
                return -score
                
            res = minimize(objective, current_points, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                candidate = res.x
                cand_score = _calc_true_score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate.copy()
                current_points = candidate
            else:
                break # Move to next init or beta? If it fails, likely stuck.

    if best_points is None:
        return inits[0]
        
    return best_points
