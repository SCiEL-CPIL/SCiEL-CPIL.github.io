    # Extract final positions
    final_pos = points[best_indices]
    
    # Center the configuration
    # Use the center of the bounding box of projections for the outer hexagon center
    projs = final_pos @ normals.T
    min_projs = np.min(projs, axis=0)
    max_projs = np.max(projs, axis=0)
    
    # The center of the enclosing hexagon in Cartesian coords
    # This is tricky to invert from projections directly without solving linear system.
    # Easier: Just shift by the centroid of the points?
    # Or shift by the mean of min/max projections mapped back?
    # For a regular hexagon, the center is the average of the 6 vertices.
    # The center of the minimal enclosing hexagon is the center of the "bounding box" in the 3 axes.
    
    # Simplest robust approach:
    # 1. Compute center C such that the max width is minimized? No, width is fixed by the set.
    # 2. The optimal center for a regular hexagon enclosing a set of points is the center of the minimal enclosing circle? No.
    # 3. It is the point that minimizes the max distance to the sides.
    
    # Let's just use the centroid of the 12 points as the center of the outer hexagon.
    # Then shift inner points by -centroid.
    centroid = np.mean(final_pos, axis=0)
    shifted_pos = final_pos - centroid
    
    # Recalculate S for shifted_pos (should be same as before)
    projs_shifted = shifted_pos @ normals.T
    widths = np.max(projs_shifted, axis=0) - np.min(projs_shifted, axis=0)
    S = float(np.max(widths) / math.sqrt(3))
    
    inner = []
    for p in shifted_pos:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    return inner, (0.0, 0.0), S, 0.0
