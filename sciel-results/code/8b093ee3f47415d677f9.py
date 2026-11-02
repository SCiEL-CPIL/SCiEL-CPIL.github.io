    # Compute pure S for best_pos
    cx = best_pos[:, 0:1]
    cy = best_pos[:, 1:2]
    vx = cx + v_x
    vy = cy + v_y
    s1 = np.abs(vx)
    s2 = np.abs(vx + vy/sqrt3)
    s3 = np.abs(vx - vy/sqrt3)
    pure_s = np.max(np.maximum(s1, np.maximum(s2, s3)))
    
    # Check disjointness
    dists = np.sqrt(np.sum((best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :])**2, axis=2))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    min_dist = np.min(dists[mask])
    
    if min_dist < sqrt3 - 1e-5:
         # Fall back to initial
         inner = [[p[0], p[1], 0.0] for p in initial_pos]
         # Compute S for initial
         cx = initial_pos[:, 0:1]
         cy = initial_pos[:, 1:2]
         vx = cx + v_x
         vy = cy + v_y
         s1 = np.abs(vx)
         s2 = np.abs(vx + vy/sqrt3)
         s3 = np.abs(vx - vy/sqrt3)
         pure_s = np.max(np.maximum(s1, np.maximum(s2, s3)))
    else:
         inner = [[best_pos[i, 0], best_pos[i, 1], 0.0] for i in range(N)]

    return inner, (0.0, 0.0), pure_s, 0.0
