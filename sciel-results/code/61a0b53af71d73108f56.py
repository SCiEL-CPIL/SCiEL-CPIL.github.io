import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    def compute_energy(centers_flat, r):
        """
        Compute energy for a given radius r.
        centers_flat: array of shape (3*N,)
        """
        c = centers_flat.reshape(N, 3)
        
        # Check if any center is outside the cube.
        # If so, energy is huge.
        # The minimizer will be constrained to box, so this shouldn't happen if bounds are set.
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, 0)
        
        # We need d_ij >= 2r
        # Penalty: (max(0, 2r - d_ij))^2
        # d_ij = sqrt(dist2)
        # To avoid sqrt for all, we can check dist2 < (2r)^2
        # But the penalty is on the distance, not squared distance.
        # Let's just compute distances. N=50 is small.
        dist = np.sqrt(dist2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dist_pairs = dist[mask]
        
        # Penalty for pairs
        # d < 2r -> (2r - d)^2
        violations = np.maximum(0, 2*r - dist_pairs)
        pair_energy = np.sum(violations * violations)
        
        # Wall distances
        # d_wall = 0.5 - max(|cx|, |cy|, |cz|)
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        d_walls = 0.5 - max_coord
        
        # Penalty for walls
        # d_wall < r -> (r - d_wall)^2
        wall_violations = np.maximum(0, r - d_walls)
        wall_energy = np.sum(wall_violations * wall_violations)
        
        return pair_energy + wall_energy

    def minimize_for_radius(r, initial_centers):
        """
        Try to find a configuration for radius r starting from initial_centers.
        Returns (success, best_centers, min_energy)
        """
        x0 = initial_centers.flatten()
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        # Use L-BFGS-B
        # We don't have an analytic gradient, so let's use Nelder-Mead or BFGS with numerical gradient?
        # Nelder-Mead is derivative-free and often works well for this.
        # However, Nelder-Mead can be slow in high dimensions (150).
        # L-BFGS-B with finite difference gradient might be better.
        
        # Let's try Nelder-Mead first as it's robust for non-smooth functions.
        # The energy function is continuous but not differentiable at the boundaries of the max(0, ...) terms.
        
        result = minimize(
            compute_energy, 
            x0, 
            args=(r,), 
            method='Nelder-Mead', 
            bounds=bounds, # Nelder-Mead ignores bounds in scipy? 
                           # Actually, scipy's Nelder-Mead does NOT support bounds.
                           # We must use L-BFGS-B or COBYLA.
                           # COBYLA is slow. L-BFGS-B is fast.
                           # Let's use L-BFGS-B.
            options={'maxiter': 500, 'disp': False}
        )
        
        # If L-BFGS-B fails or gets stuck, we might need a different method.
        # Let's stick to L-BFGS-B for speed.
        
        min_energy = result.fun
        best_centers = result.x.reshape(N, 3)
        
        # Check if energy is effectively zero
        # Threshold depends on r.
        success = min_energy < 1e-8
        
        return success, best_centers, min_energy

    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    current_centers = grid[:N].copy()
    
    # Binary search on radius
    r_low = 0.125  # Grid score
    r_high = 0.140
    
    best_centers = current_centers.copy()
    
    while time.time() - start_time < timeout and (r_high - r_low) > 1e-5:
        r_mid = (r_low + r_high) / 2.0
        
        # Try to minimize energy for r_mid
        # We should start from the current best centers to maintain continuity
        success, new_centers, energy = minimize_for_radius(r_mid, current_centers)
        
        if success:
            r_low = r_mid
            current_centers = new_centers
            best_centers = new_centers
        else:
            r_high = r_mid
            # If it fails, we don't update current_centers? 
            # Or we update to the best found in the minimization to help the next try?
            # Usually, it's better to keep the last valid configuration or the one that minimized energy the most.
            # Let's update current_centers to the result of the minimization even if not fully successful,
            # as it might be closer to a feasible point for slightly smaller r.
            current_centers = new_centers
            
    return best_centers
# EVOLVE-BLOCK-END
