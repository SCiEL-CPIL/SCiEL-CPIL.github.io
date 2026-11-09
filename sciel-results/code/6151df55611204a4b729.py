def _force_optimize(points, time_limit=60.0):
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    lr = 0.005 # Start small
    decay = 0.9995
    
    for step in range(20000):
        if time.time() - start_time > time_limit:
            break
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2) + 1e-8
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        inv_dist_sq[dist_sq == 1.0] = 0
        
        weights = (inv_dist_sq**2)[:, :, np.newaxis]
        forces = 2.0 * np.sum(diff * weights, axis=1)
        
        norms = np.linalg.norm(pts, axis=1)
        mask_outside = norms > 1.0
        pen_grad = np.zeros_like(pts)
        if np.any(mask_outside):
            scale = 2.0 * (norms[mask_outside] - 1.0) / norms[mask_outside]
            pen_grad[mask_outside] = -scale[:, np.newaxis] * pts[mask_outside]
            
        total_force = forces + pen_grad
        
        pts += lr * total_force
        
        lr *= decay
        
    return pts
