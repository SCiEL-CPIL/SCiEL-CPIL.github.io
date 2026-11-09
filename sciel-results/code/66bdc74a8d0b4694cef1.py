def _calc_min_enclosing_side_fast(centers, angles):
    verts = []
    for i in range(len(centers)):
        v = _hex_verts(centers[i][0], centers[i][1], 1.0, angles[i])
        verts.append(v)
    all_verts = np.vstack(verts)
    centroid = np.mean(centers, axis=0)
    all_verts_rel = all_verts - centroid
    
    # Sample angles
    angle_samples = np.arange(0, 60, 2.0) # 30 samples
    thetas = np.radians(angle_samples)
    
    # Normals for all samples at once
    # edge_angles = theta + 30 deg + 60 deg * k
    # Shape: (30, 6, 2)
    k = np.arange(6)
    edge_angles = thetas[:, np.newaxis] + np.radians(30) + k[np.newaxis, :] * np.pi / 3
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)]) # (30, 12) -> reshape to (30, 6, 2)
    normals = normals.reshape(len(angle_samples), 6, 2)
    
    # Projections: (72, 30, 6)
    # dot(v, n) for each v, each angle, each normal
    # v: (72, 1, 1), n: (1, 30, 6, 2)
    # This might be memory heavy. 72*30*6 = 12960 floats. Fine.
    projections = np.einsum('ij,klj->ikl', all_verts_rel, normals)
    
    # Max over vertices (i)
    max_projections = np.max(projections, axis=0) # (30, 6)
    
    # Max over normals (l) for each angle (k)
    max_per_angle = np.max(max_projections, axis=1) # (30,)
    
    # Find best angle
    min_idx = np.argmin(max_per_angle)
    min_side = max_per_angle[min_idx] / (np.sqrt(3) / 2)
    best_angle = angle_samples[min_idx]
    
    # Fine search around best angle
    fine_samples = np.arange(best_angle - 2, best_angle + 2, 0.5)
    if len(fine_samples) > 0:
        thetas_fine = np.radians(fine_samples)
        edge_angles_fine = thetas_fine[:, np.newaxis] + np.radians(30) + k[np.newaxis, :] * np.pi / 3
        normals_fine = np.column_stack([np.cos(edge_angles_fine), np.sin(edge_angles_fine)])
        normals_fine = normals_fine.reshape(len(fine_samples), 6, 2)
        projections_fine = np.einsum('ij,klj->ikl', all_verts_rel, normals_fine)
        max_projections_fine = np.max(projections_fine, axis=0)
        max_per_angle_fine = np.max(max_projections_fine, axis=1)
        min_idx_fine = np.argmin(max_per_angle_fine)
        if max_per_angle_fine[min_idx_fine] < max_per_angle[min_idx]:
            min_side = max_per_angle_fine[min_idx_fine] / (np.sqrt(3) / 2)
            best_angle = fine_samples[min_idx_fine]
            
    return min_side, best_angle, centroid
