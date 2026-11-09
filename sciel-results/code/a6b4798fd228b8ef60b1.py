<<<<<<< SEARCH
def _local_refine_combined(pts, deadline, max_iters=3000, initial_step=0.05, 
                           beta_smooth=50.0, beta_dual=10.0):
    """
    Local refinement combining smooth gradient and dual uniformization.
    Uses smooth gradient as primary, dual as fallback when smooth stalls.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    smooth_stall_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Primary: smooth gradient step
        new_pts_smooth = _smooth_gradient_step(pts, step_size, beta=beta_smooth, grid_size=2000)
        new_radius_smooth = _compute_covering_radius(new_pts_smooth)
        
        if new_radius_smooth < best_radius - 1e-12:
            best_radius = new_radius_smooth
            best_pts = new_pts_smooth.copy()
            pts = new_pts_smooth
            no_improve_count = 0
            smooth_stall_count = 0
            step_size *= 0.995
        else:
            # Try dual uniformization as fallback
            new_pts_dual = _dual_uniformization_step(pts, step_size, beta=beta_dual)
            new_radius_dual = _compute_covering_radius(new_pts_dual)
            
            if new_radius_dual < best_radius - 1e-12:
                best_radius = new_radius_dual
                best_pts = new_pts_dual.copy()
                pts = new_pts_dual
                no_improve_count = 0
                smooth_stall_count += 1
                step_size *= 0.995
            else:
                no_improve_count += 1
                smooth_stall_count += 1
                step_size *= 0.9
                
                if step_size < 1e-6 or no_improve_count > 50:
                    break
    
    return best_pts, best_radius
=======
def _local_refine_combined(pts, deadline, max_iters=3000, initial_step=0.05, 
                           beta_smooth=50.0, beta_dual=10.0):
    """
    Local refinement combining smooth gradient and dual uniformization.
    Uses smooth gradient as primary, dual as fallback when smooth stalls.
    Adds line search for better step acceptance.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Try multiple step sizes with line search
        improved = False
        for trial in range(3):
            s = step_size * (0.5 ** trial)
            if s < 1e-8:
                break
                
            # Primary: smooth gradient step
            new_pts_smooth = _smooth_gradient_step(pts, s, beta=beta_smooth, grid_size=2000)
            new_radius_smooth = _compute_covering_radius(new_pts_smooth)
            
            if new_radius_smooth < best_radius - 1e-13:
                best_radius = new_radius_smooth
                best_pts = new_pts_smooth.copy()
                pts = new_pts_smooth
                no_improve_count = 0
                step_size = s
                improved = True
                break
        
        if not improved:
            # Try dual uniformization as fallback
            new_pts_dual = _dual_uniformization_step(pts, step_size, beta=beta_dual)
            new_radius_dual = _compute_covering_radius(new_pts_dual)
            
            if new_radius_dual < best_radius - 1e-13:
                best_radius = new_radius_dual
                best_pts = new_pts_dual.copy()
                pts = new_pts_dual
                no_improve_count = 0
                step_size *= 0.995
            else:
                no_improve_count += 1
                step_size *= 0.85
                
                if step_size < 1e-7 or no_improve_count > 40:
                    break
    
    return best_pts, best_radius
>>>>>>> REPLACE

