<<<<<<< SEARCH
    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
=======
    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    # Local Search (Jitter) to refine positions
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        best_pts = best_global_pts.copy()
        best_S = best_global_S
        best_beta = best_global_beta
        best_center = best_global_center
        angle_deg = math.degrees(best_beta)
        
        # We assume all inner hexagons have the same rotation angle_deg
        # We will jitter their positions
        
        for iter in range(200):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            i = np.random.randint(len(best_pts))
            dx = np.random.uniform(-0.15, 0.15)
            dy = np.random.uniform(-0.15, 0.15)
            
            new_pts = best_pts.copy()
            new_pts[i, 0] += dx
            new_pts[i, 1] += dy
            
            # Check overlap with all others
            valid = True
            for j in range(len(new_pts)):
                if i == j:
                    continue
                # Since all hexagons have the same rotation, we can check overlap
                # using the relative positions. The angle is the same for both.
                if _hex_overlap((new_pts[i,0], new_pts[i,1]), (new_pts[j,0], new_pts[j,1]), angle_deg, angle_deg):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate new S with fixed beta and center (fast approximation)
            S_new, _, _ = calc_S_for_beta(new_pts, best_beta)
            
            if S_new < best_S:
                best_S = S_new
                best_pts = new_pts
                
                # Re-optimize outer hexagon periodically
                if iter % 20 == 0:
                    S_opt, beta_opt, center_opt = optimize_orientation_only(best_pts)
                    if S_opt < best_S:
                        best_S = S_opt
                        best_beta = beta_opt
                        best_center = center_opt
                        angle_deg = math.degrees(best_beta)

        if best_S < best_global_S:
            best_global_S = best_S
            best_global_pts = best_pts
            best_global_beta = best_beta
            best_global_center = best_center
>>>>>>> REPLACE
