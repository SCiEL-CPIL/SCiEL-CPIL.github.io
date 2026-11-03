<<<<<<< SEARCH
    def newton_polish_center(pts, beta):
        """High-precision center polish using bisection on active constraint sets.
        At optimum, the max projection D is determined by a small set of active
        points. We find the center that equalizes the extreme projections."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Get initial center from LP
        S_init, center_init, _ = calc_S_for_beta(pts, beta)
        if center_init is None:
            return None, None
        
        cx, cy = center_init
        cur_S = exact_S(pts, cx, cy, beta)
        
        for iteration in range(30):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T  # (n_pts, 6)
            D = np.max(np.abs(projs))
            
            # Find all active (point, direction) pairs
            active_pos = []  # (point_idx, normal_idx) for positive
            active_neg = []  # (point_idx, normal_idx) for negative
            for k in range(6):
                for i in range(len(pts)):
                    if projs[i, k] >= D - 1e-12:
                        active_pos.append((i, k))
                    if projs[i, k] <= -D + 1e-12:
                        active_neg.append((i, k))
            
            if not active_pos and not active_neg:
                break
            
            # The optimal center satisfies: for the active set, the extreme
            # projections are equal. We use a 2D bisection/Newton approach.
            # Try small perturbations and keep the best.
            best_cx, best_cy, best_S = cx, cy, cur_S
            for scale in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                    ncx = cx + dx * scale
                    ncy = cy + dy * scale
                    S_try = exact_S(pts, ncx, ncy, beta)
                    if S_try < best_S - 1e-15:
                        best_S = S_try
                        best_cx = ncx
                        best_cy = ncy
            
            if abs(best_S - cur_S) < 1e-16:
                break
            cx, cy = best_cx, best_cy
            cur_S = best_S
        
        return (cx, cy), cur_S
=======
    def newton_polish_center(pts, beta):
        """High-precision center polish using multi-scale local search.
        At optimum, the max projection D is determined by a small set of active
        points. We iteratively refine the center."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Get initial center from LP
        S_init, center_init, _ = calc_S_for_beta(pts, beta)
        if center_init is None:
            return None, None
        
        cx, cy = center_init
        cur_S = exact_S(pts, cx, cy, beta)
        
        # Multi-scale refinement with decreasing step sizes
        scales = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1),
                      (0.707,0.707),(-0.707,0.707),(0.707,-0.707),(-0.707,-0.707)]
        
        for scale in scales:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            improved = True
            while improved:
                improved = False
                best_cx, best_cy, best_S = cx, cy, cur_S
                for dx, dy in directions:
                    ncx = cx + dx * scale
                    ncy = cy + dy * scale
                    S_try = exact_S(pts, ncx, ncy, beta)
                    if S_try < best_S - 1e-15:
                        best_S = S_try
                        best_cx = ncx
                        best_cy = ncy
                        improved = True
                if improved:
                    cx, cy = best_cx, best_cy
                    cur_S = best_S
        
        return (cx, cy), cur_S
>>>>>>> REPLACE

