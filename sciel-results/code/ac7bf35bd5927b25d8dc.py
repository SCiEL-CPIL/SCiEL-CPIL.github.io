<<<<<<< SEARCH
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
=======
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_beta = 0.0
        S, best_global_center, _ = exact_S_for_beta(best_global_pts, 0.0)
        best_global_S = S

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Recompute exact enclosing hexagon side using Minkowski-sum center
    normals = np.array([
        [math.cos(best_global_beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(best_global_beta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])
    projs = normals @ best_global_pts.T
    max_projs = projs.max(axis=1)
    min_projs = projs.min(axis=1)
    half_widths = 0.5 * (max_projs - min_projs)
    D_final = float(np.max(half_widths))
    S_final = (D_final + APOTHEM) / APOTHEM
    
    # Small safety margin
    S_out = S_final * (1.0 + 1e-10)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_out)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
>>>>>>> REPLACE
