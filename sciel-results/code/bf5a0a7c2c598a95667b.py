<<<<<<< SEARCH
    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12, math.pi/6]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # Final high-precision polish using exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0.0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = abs((px - cx) * nx + (py - cy) * ny)
                            if proj > max_proj:
                                max_proj = proj
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center
=======
    def optimize_fixed_positions(pts):
        """Optimize outer hexagon (center + rotation) for fixed inner positions."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi/6),
                                     method='bounded', options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                for temp in [0.1, 0.03, 0.01]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                        return S
                    init = [best_center[0], best_center[1], best_beta]
                    res = minimize(smooth_obj, init, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res.fun < 1e10:
                        beta_r = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_beta = beta_r
                            best_center = c_exact
        except Exception:
            pass
        
        # Exact polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = np.array([
                        (math.cos(beta + math.pi/6.0 + k*math.pi/3.0),
                         math.sin(beta + math.pi/6.0 + k*math.pi/3.0))
                        for k in range(6)
                    ])
                    centered = pts - np.array([cx, cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    return (max_proj + APOTHEM) / APOTHEM
                
                res = minimize(objective, [best_center[0], best_center[1], best_beta],
                             method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    def hexagon_overlap_check(pts, beta, min_dist=2.0 - 1e-6):
        """Check if any two unit hexagons at given positions/rotations overlap.
        Uses vertex-to-vertex and edge-to-vertex distance as proxy."""
        n = len(pts)
        # For two unit hexagons with same rotation, minimum center distance for no overlap is 2*APOTHEM*2 = 2*sqrt(3)
        # Actually for two unit hexagons (same orientation), they touch at distance 2*sqrt(3) center-to-center
        # For different rotations, minimum distance can be smaller.
        # Conservative: use 2*APOTHEM*2 = 2*sqrt(3) ≈ 3.464 for same-orientation
        # For arbitrary rotation, the minimum non-overlap distance is 2 (vertex to vertex of opposite hex)
        # Actually the minimum distance between centers of two non-overlapping unit hexagons
        # is 2*APOTHEM = sqrt(3) ≈ 1.732 (when a flat side of one touches a flat side of other)
        # Wait: apothem = sqrt(3)/2. Two hexagons with parallel faces touch at distance 2*apothem = sqrt(3).
        # But they can also touch vertex-to-vertex at distance 2 (side length).
        # The minimum center-to-center distance for non-overlap depends on relative rotation.
        # For safety, use sqrt(3) as the minimum.
        min_center_dist = math.sqrt(3) - 1e-4
        for i in range(n):
            for j in range(i+1, n):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_center_dist:
                    return False
        return True

    def force_directed_layout(seed_pts, n_iter=500, beta=0.0):
        """Optimize positions continuously using force-directed method.
        Minimize enclosing hexagon size while maintaining separation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        pts = seed_pts.astype(float).copy()
        n = len(pts)
        
        # Center the points
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        
        min_sep = math.sqrt(3) - 1e-5  # minimum center distance
        lr = 0.02
        lr_decay = 0.995
        
        best_S = float('inf')
        best_pts = pts.copy()
        best_center = (0.0, 0.0)
        
        for it in range(n_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute enclosing hexagon for current positions
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_pts = pts.copy()
                best_center = center
            
            if not np.isfinite(S):
                break
            
            cx, cy = center if center else (0.0, 0.0)
            
            # Gradient of S w.r.t. each point (numerical)
            normals = np.array([
                (math.cos(beta + math.pi/6.0 + k*math.pi/3.0),
                 math.sin(beta + math.pi/6.0 + k*math.pi/3.0))
                for k in range(6)
            ])
            
            # For each point, compute which face is binding
            forces = np.zeros_like(pts)
            centered = pts - np.array([cx, cy])
            projs = np.abs(centered @ normals.T)  # (n, 6)
            
            # The binding face for each point
            binding_face = np.argmax(projs, axis=1)
            binding_proj = projs[np.arange(n), binding_face]
            
            # Force: push points that are on the boundary inward
            for i in range(n):
                nx, ny = normals[binding_face[i]]
                sign = 1.0 if (centered[i] @ normals[binding_face[i]]) > 0 else -1.0
                forces[i, 0] -= sign * nx * 0.1
                forces[i, 1] -= sign * ny * 0.1
            
            # Separation forces (repulsion between too-close hexagons)
            for i in range(n):
                for j in range(i+1, n):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-10
                    if dist < min_sep * 1.5:
                        # Repulsive force
                        strength = (min_sep * 1.5 - dist) / (min_sep * 1.5)
                        fx = strength * dx / dist * 0.5
                        fy = strength * dy / dist * 0.5
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Update positions
            pts += lr * forces
            lr *= lr_decay
            
            # Hard constraint: ensure minimum separation
            for i in range(n):
                for j in range(i+1, n):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-10
                    if dist < min_sep:
                        correction = (min_sep - dist) / dist * 0.5
                        pts[i, 0] += correction * dx
                        pts[i, 1] += correction * dy
                        pts[j, 0] -= correction * dx
                        pts[j, 1] -= correction * dy
        
        return best_pts, best_S, best_center
>>>>>>> REPLACE

