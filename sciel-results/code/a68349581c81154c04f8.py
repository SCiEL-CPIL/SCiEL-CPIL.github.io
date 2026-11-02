<<<<<<< SEARCH
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_positions = None
    best_global_angles = None

    # First pass: quick evaluation to find top candidates
    quick_results = []
    for i, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        best_quick_S = float('inf')
        best_quick_beta = 0.0
        best_quick_center = None
        for j in range(13):
            beta = j * (math.pi / 6) / 12
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_quick_S:
                best_quick_S = S
                best_quick_beta = beta
                best_quick_center = center
        quick_results.append((best_quick_S, i, pts, best_quick_beta, best_quick_center))
    
    quick_results.sort(key=lambda x: x[0])
    
    # Phase 1: Lattice-based optimization (top 8 candidates)
    max_detailed = min(len(quick_results), 8)
    for idx in range(max_detailed):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        quick_S, axial_idx, pts, quick_beta, quick_center = quick_results[idx]
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

    # Phase 2: Continuous NLP refinement on top candidates
    max_continuous = min(len(quick_results), 6)
    for idx in range(max_continuous):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        quick_S, axial_idx, pts, quick_beta, quick_center = quick_results[idx]
        
        S_cont, positions, angles, center_beta = continuous_refine(pts, quick_beta, quick_center[0], quick_center[1])
        if S_cont < best_global_S:
            best_global_S = S_cont
            best_global_pts = np.array(positions)
            best_global_beta = center_beta[2]
            best_global_center = (center_beta[0], center_beta[1])
            best_global_positions = positions
            best_global_angles = angles
=======
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_positions = None
    best_global_angles = None

    # First pass: quick evaluation to find top candidates
    quick_results = []
    for i, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        best_quick_S = float('inf')
        best_quick_beta = 0.0
        best_quick_center = None
        for j in range(13):
            beta = j * (math.pi / 6) / 12
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_quick_S:
                best_quick_S = S
                best_quick_beta = beta
                best_quick_center = center
        quick_results.append((best_quick_S, i, pts, best_quick_beta, best_quick_center))
    
    quick_results.sort(key=lambda x: x[0])
    
    # Phase 1: Lattice-based optimization (top 8 candidates)
    max_detailed = min(len(quick_results), 8)
    for idx in range(max_detailed):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        quick_S, axial_idx, pts, quick_beta, quick_center = quick_results[idx]
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            # Store the lattice positions and angles (all 0) for fallback
            best_global_positions = [tuple(p) for p in pts]
            best_global_angles = [0.0] * N

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
        best_global_positions = [tuple(p) for p in best_global_pts]
        best_global_angles = [0.0] * N

    # Phase 2: Force-directed refinement on top candidates
    max_continuous = min(len(quick_results), 6)
    for idx in range(max_continuous):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        quick_S, axial_idx, pts, quick_beta, quick_center = quick_results[idx]
        
        # Try to improve the packing using force-directed layout
        fd_result = force_directed_pack(pts)
        if fd_result is not None:
            positions, angles = fd_result
            # Now find the minimal enclosing hexagon for this new configuration
            # We can use a simplified version of optimize_shape or just calc_S_exact with a few betas
            best_fd_S = float('inf')
            best_fd_beta = quick_beta
            best_fd_center = quick_center
            
            # Try a few betas
            for j in range(13):
                beta = j * (math.pi / 6) / 12
                S, center, D = calc_S_for_beta(np.array(positions), beta)
                if S < best_fd_S:
                    best_fd_S = S
                    best_fd_beta = beta
                    best_fd_center = center
            
            if best_fd_S < best_global_S:
                best_global_S = best_fd_S
                best_global_pts = np.array(positions)
                best_global_beta = best_fd_beta
                best_global_center = best_fd_center
                best_global_positions = positions
                best_global_angles = angles
>>>>>>> REPLACE

