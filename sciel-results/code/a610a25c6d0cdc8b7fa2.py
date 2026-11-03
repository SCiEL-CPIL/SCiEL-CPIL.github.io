"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the max valid radius for given centers."""
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary_dist = np.min(dist_to_boundary)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _relax(centers, target_r, max_iter=200):
    """
    Relax centers to satisfy non-overlap and boundary constraints for target_r.
    Minimizes the sum of squared violations.
    Returns relaxed centers and the final violation value.
    """
    x0 = centers.flatten()
    
    def violation(x):
        c = x.reshape(-1, 2)
        # Pairwise violations
        dists = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Overlap if dist < 2*target_r
        overlaps = np.maximum(0, 2.0 * target_r - dists)
        pair_violation = np.sum(overlaps**2)
        
        # Boundary violations
        # dist_to_boundary < target_r
        dist_to_boundary = np.min(0.5 - np.abs(c), axis=1)
        bound_violations = np.maximum(0, target_r - dist_to_boundary)
        bound_violation = np.sum(bound_violations**2)
        
        return pair_violation + bound_violation
    
    def violation_grad(x):
        # Gradient of violation
        c = x.reshape(-1, 2)
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        dists = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlaps = np.maximum(0, 2.0 * target_r - dists)
        
        # d/dx_i (max(0, 2r - dist_ij)^2) = 2 * max(0, 2r - dist_ij) * (-1) * (x_i - x_j) / dist_ij
        # Note: dist_ij = |x_i - x_j|
        # grad_i = -2 * overlap_ij * (x_i - x_j) / dist_ij
        # grad_j = +2 * overlap_ij * (x_i - x_j) / dist_ij
        
        # Vectorized calculation
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # We need to sum contributions to each point
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-12, dists, 1.0)
        
        # Factor for each pair (i,j)
        # factor_ij = 2 * overlap_ij / dist_ij
        factor = 2.0 * overlaps / dists_safe
        # Set diagonal to 0
        np.fill_diagonal(factor, 0)
        
        # grad_i = - sum_j factor_ij * diff_ij
        # diff_ij = c_i - c_j
        # sum over j
        grad_pairs = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary gradient
        # d/dx_i (max(0, r - (0.5 - |x_i|))^2)
        # Let d_i = 0.5 - |x_i|
        # viol_i = max(0, r - d_i)
        # grad_x_i = 2 * viol_i * (-d/dx_i d_i)
        # d_i = min(0.5 - x_i, 0.5 + x_i) ... actually min(0.5 - |x_i|, 0.5 - |y_i|)
        # This is non-smooth. We can use subgradient or just use Powell/NM which doesn't need gradient.
        # Given the small N, Powell or NM is safer.
        pass
        
    # Use Powell or Nelder-Mead as gradient is tricky with min/max
    result = minimize(violation, x0, method='Powell', options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12})
    
    relaxed_centers = result.x.reshape(-1, 2)
    final_violation = result.fun
    
    return relaxed_centers, final_violation

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_r = 0.0
    best_centers = None
    
    # Initial grid setup
    # 8x8 grid = 64 points, take 50
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    base_centers = grid[:N].copy()
    
    # Try multiple restarts with jitter
    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        # Jitter initialization
        # Small jitter to break symmetry
        jitter_scale = 0.01
        centers = base_centers + np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
        
        # Ensure centers are within bounds initially (though jitter is small)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Growth loop
        r = 0.05
        # We will try to grow r
        # If relaxation fails (violation > threshold), we stop growing for this restart
        # and record the best valid radius found so far in this restart
        
        current_best_r_for_restart = _compute_radius(centers)
        
        for step in range(50):
            if time.time() > deadline - 5:
                break
                
            # Relax for current r
            relaxed_centers, viol = _relax(centers, r, max_iter=100)
            
            # Check if relaxation was successful
            # If viol is small, the packing is valid for r
            if viol < 1e-8:
                # Compute actual max radius for this configuration
                actual_r = _compute_radius(relaxed_centers)
                
                if actual_r < r - 1e-6:
                    # Something went wrong, maybe numerical issue
                    # Accept actual_r as the limit
                    if actual_r > current_best_r_for_restart:
                        current_best_r_for_restart = actual_r
                    break
                else:
                    # We can grow
                    # Grow to actual_r or a bit less to be safe
                    new_r = actual_r
                    if new_r <= r + 1e-6:
                        # Converged
                        if actual_r > current_best_r_for_restart:
                            current_best_r_for_restart = actual_r
                        break
                    else:
                        # Update centers and r
                        centers = relaxed_centers
                        r = new_r
                        continue
            else:
                # Relaxation failed to eliminate overlaps for this r
                # This means r is too large for this configuration
                # We can try to back off slightly? Or just accept current best for this restart
                # Let's try to relax with a slightly smaller r to see if we can recover?
                # No, better to just stop. The current configuration is the best we can get for this path.
                # But wait, we might have a valid packing for a smaller r.
                # The previous iteration's centers were valid for the previous r.
                # We should keep the last valid state.
                
                # Actually, if _relax fails, it returns the best effort.
                # We should compute the radius of the relaxed_centers even if viol > 0, 
                # but it's likely invalid. 
                # Better to use the centers from the last successful iteration.
                # However, we didn't store them. 
                # Let's just break. The current `centers` variable holds the last valid ones?
                # No, `centers` was updated to `relaxed_centers` only if we continued.
                # In the `else` block, we did NOT update `centers`.
                # So `centers` is still the previous valid configuration.
                # `r` is the target that failed.
                # So the current `centers` is valid for `r` (approximately, since it was the result of a successful relax for `r` in the previous step? 
                # No, if step 1 succeeded, centers became C1, r became R1.
                # Step 2 tries R1. If it fails, centers is still C1. R1 is the max valid for C1? 
                # Yes, because we grew R to actual_r of C1. So C1 is valid for R1.
                # So we can just break.
                
                break
        
        # After growth loop, update global best
        final_r = _compute_radius(centers)
        if final_r > best_r:
            best_r = final_r
            best_centers = centers.copy()
            
        # If we are getting close to the known best, we might want to do a final polish
        if best_r > 0.0713:
             # Optional: Run a final high-precision relaxation on the best found
             pass

    if best_centers is None:
        return base_centers
    return best_centers
# EVOLVE-BLOCK-END
