def _optimize_from_start(pts0, time_budget):
    best_pts = pts0.copy()
    best_energy = _log_energy_and_grad(pts0)[0]
    
    pts = best_pts.copy()
    lr = 0.1
    max_iters = 5000
    start_t = time.time()
    
    for i in range(max_iters):
        if time.time() > time_budget:
            break
        
        e, g = _log_energy_and_grad(pts)
        
        # Tangential gradient
        grad_tan = g - (g * pts).sum(axis=1, keepdims=True) * pts
        
        # Check convergence
        if np.linalg.norm(grad_tan) < 1e-12:
            break
            
        # Backtracking line search
        accepted = False
        for _ in range(50):
            # Step
            pts_new = pts - lr * grad_tan
            pts_new = _project_to_sphere(pts_new)
            
            e_new, _ = _log_energy_and_grad(pts_new)
            
            if e_new < e:
                pts = pts_new
                accepted = True
                break
            lr *= 0.5
            
        if not accepted:
            # Try to recover if stuck?
            break
            
        if e_new < best_energy:
            best_pts = pts.copy()
            best_energy = e_new
            
    return best_pts, best_energy
