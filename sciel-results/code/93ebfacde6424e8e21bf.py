<<<<<<< SEARCH
def _feasible(pts, r, max_iter=500, deadline=None):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses force-directed relaxation (projected gradient descent).
    Returns (success, pts).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    # Adaptive step size
    step = 0.1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances and forces
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        
        # 1. Pairwise Repulsion Forces
        # We want to push apart pairs where dist < 2r
        overlap = (dist < 2 * r) & (dist > 1e-12)
        # Force magnitude proportional to overlap
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        # Normalize directions
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)  # (n, 2)
        
        # 2. Wall Constraints
        # Push away from walls if pts[i,k] + r > HALF or pts[i,k] - r < -HALF
        wall_force = np.zeros_like(pts)
        for k in range(2):
            # Right/Top wall
            too_right = pts[:, k] > HALF - r
            wall_force[too_right, k] = -(pts[too_right, k] - (HALF - r))
            # Left/Bottom
            too_left = pts[:, k] < -HALF + r
            wall_force[too_left, k] = (pts[too_left, k] - (-HALF + r))
            
        grad += wall_force
        
        # Update positions
        new_pts = pts + step * grad
        
        # Project back to valid bounds (center must be in [-HALF+r, HALF-r])
        new_pts = np.clip(new_pts, -HALF + r, HALF - r)
        
        # Check convergence / success
        # Recompute distances for new_pts
        diff_new = new_pts[:, None, :] - new_pts[None, :, :]
        dist_new = np.sqrt(np.sum(diff_new * diff_new, axis=2))
        np.fill_diagonal(dist_new, np.inf)
        min_pair_new = np.min(dist_new)
        min_wall_new = np.min(HALF - np.abs(new_pts))
        
        if min_pair_new >= 2 * r - 1e-10 and min_wall_new >= r - 1e-10:
            return True, new_pts
            
        # Adaptive step size reduction if we are oscillating or not improving
        # Simple heuristic: reduce step if min_pair_new is not significantly larger than min_pair
        # For simplicity, we just keep a constant step but maybe reduce it slightly over time
        # or use a fixed small step if we are close.
        # Let's try a simple decay
        if it % 50 == 0:
             step *= 0.9
             
        pts = new_pts
        
    # Final check
    dist_final = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist_final, np.inf)
    min_pair_final = np.min(dist_final)
    min_wall_final = np.min(HALF - np.abs(pts))
    success = (min_pair_final >= 2 * r - 1e-10) and (min_wall_final >= r - 1e-10)
    return success, pts
=======
def _objective(x, r):
    """
    Objective function for the optimizer.
    Minimizes the sum of squared violations of constraints.
    Violations:
    1. Circles overlapping: dist < 2r -> (2r - dist)^2
    2. Circles outside bounds: coord > HALF-r or coord < -HALF+r
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Overlap violations
    overlap_viol = np.maximum(0, 2 * r - dist)
    cost_overlap = np.sum(overlap_viol ** 2)
    
    # Boundary violations
    # x-coords
    x_viol = np.maximum(0, pts[:, 0] - (HALF - r)) + np.maximum(0, (-HALF + r) - pts[:, 0])
    # y-coords
    y_viol = np.maximum(0, pts[:, 1] - (HALF - r)) + np.maximum(0, (-HALF + r) - pts[:, 1])
    
    cost_bound = np.sum(x_viol ** 2) + np.sum(y_viol ** 2)
    
    return cost_overlap + cost_bound

def _optimize_config(seed_pts, r, deadline=None):
    """
    Optimize the configuration starting from seed_pts to satisfy constraints for radius r.
    Uses L-BFGS-B.
    Returns (success, best_pts).
    """
    x0 = seed_pts.flatten()
    
    # Bounds for the optimizer
    bounds = [(-HALF + r, HALF - r)] * (N * 2)
    
    # Check time
    if deadline is not None and time.time() > deadline:
        return False, seed_pts
    
    try:
        # Use L-BFGS-B for bounded optimization
        res = minimize(
            _objective, 
            x0, 
            args=(r,), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        best_pts = res.x.reshape(N, 2)
        val = res.fun
        
        # Success if the objective function is close to zero
        # 1e-6 squared violation is very small
        success = val < 1e-8
        
        return success, best_pts
        
    except Exception as e:
        return False, seed_pts
>>>>>>> REPLACE

