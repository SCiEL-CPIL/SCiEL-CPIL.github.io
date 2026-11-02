<<<<<<< SEARCH
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Use a very small safety margin to ensure no overlap
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
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

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
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final high-precision polish: optimize center and rotation for the best configuration
    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        pts = best_global_pts
        
        def final_objective(params):
            fcx, fcy, fbeta = params
            fbeta = ((fbeta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            normals = np.array([
                (math.cos(fbeta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(fbeta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            pts_centered = pts - np.array([fcx, fcy])
            projs = pts_centered @ normals.T
            max_proj = np.max(np.abs(projs))
            return (max_proj + APOTHEM) / APOTHEM
        
        # Multiple starting points for final polish
        polish_starts = [
            [cx, cy, best_global_beta],
            [cx, cy, best_global_beta + 0.01],
            [cx, cy, best_global_beta - 0.01],
            [cx + 0.01, cy, best_global_beta],
            [cx - 0.01, cy, best_global_beta],
            [cx, cy + 0.01, best_global_beta],
            [cx, cy - 0.01, best_global_beta],
            [cx + 0.005, cy + 0.005, best_global_beta],
            [cx - 0.005, cy - 0.005, best_global_beta],
        ]
        
        best_final_S = best_global_S
        best_final_cx = cx
        best_final_cy = cy
        best_final_beta = best_global_beta
        
        for init in polish_starts:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(final_objective, init, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < best_final_S:
                    best_final_S = res.fun
                    best_final_cx = res.x[0]
                    best_final_cy = res.x[1]
                    best_final_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass
        
        if best_final_S < best_global_S:
            best_global_S = best_final_S
            best_global_beta = best_final_beta
            cx, cy = best_final_cx, best_final_cy
            angle_deg = math.degrees(best_global_beta)

    # Use a very small safety margin to ensure no overlap
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
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
>>>>>>> REPLACE
