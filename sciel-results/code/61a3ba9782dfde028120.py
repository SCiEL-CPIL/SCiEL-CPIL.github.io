import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    def objective(pts_flat, r_target):
        """
        Objective function for minimizing overlap and wall violations.
        pts_flat: 1D array of length 300.
        r_target: target radius.
        Returns: scalar energy.
        """
        pts = pts_flat.reshape(N, 3)
        
        # Wall violations
        # Distance to wall is 0.5 - |coord|
        # Violation if r_target > 0.5 - |coord| => |coord| > 0.5 - r_target
        wall_violations = np.abs(pts) - (0.5 - r_target)
        wall_violations = np.maximum(wall_violations, 0.0)
        wall_energy = np.sum(wall_violations ** 2)
        
        # Pairwise violations
        # We need to compute pairwise distances efficiently.
        # For N=100, pdist is fast enough.
        dists = pdist(pts)
        min_dist_required = 2.0 * r_target
        pair_violations = min_dist_required - dists
        pair_violations = np.maximum(pair_violations, 0.0)
        pair_energy = np.sum(pair_violations ** 2)
        
        return wall_energy + pair_energy

    def optimize_for_radius(r_target, initial_pts, time_limit=10.0):
        """
        Attempt to find a valid configuration for r_target using L-BFGS-B.
        Returns (pts, energy).
        """
        start_time = time.time()
        
        # Flatten initial points
        x0 = initial_pts.flatten()
        
        # Bounds: points must stay within [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # We use a wrapper to stop if time limit is exceeded
        # L-BFGS-B doesn't have a direct time limit, so we might need to be careful.
        # Given N=100, L-BFGS-B should be fast.
        
        try:
            res = minimize(
                objective, 
                x0, 
                args=(r_target,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
        except Exception:
            return initial_pts, np.inf

        if time.time() - start_time > time_limit:
             # If it takes too long, we might return the current best or just fail
             # L-BFGS-B usually converges quickly or fails fast for this size.
             pass

        if res.success or res.fun < 1e-6:
            return res.x.reshape(N, 3), res.fun
        else:
            return res.x.reshape(N, 3), res.fun

    def get_initial_configs():
        """
        Generate a set of diverse initial configurations.
        """
        configs = []
        
        # 5x5x4 Grid
        nx, ny, nz = 5, 5, 4
        xs = np.linspace(-0.4, 0.4, nx)
        ys = np.linspace(-0.4, 0.4, ny)
        zs = np.linspace(-0.4, 0.4, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        configs.append(grid)
        
        # 4x5x5 Grid
        nx, ny, nz = 4, 5, 5
        xs = np.linspace(-0.4, 0.4, nx)
        ys = np.linspace(-0.4, 0.4, ny)
        zs = np.linspace(-0.4, 0.4, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        configs.append(grid)
        
        # 5x4x5 Grid
        nx, ny, nz = 5, 4, 5
        xs = np.linspace(-0.4, 0.4, nx)
        ys = np.linspace(-0.4, 0.4, ny)
        zs = np.linspace(-0.4, 0.4, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        configs.append(grid)
        
        # Jittered Grids to break symmetry
        rng = np.random.default_rng(42)
        for base_config in configs[:3]:
            jitter = rng.uniform(-0.02, 0.02, size=(N, 3))
            configs.append(base_config + jitter)
            
        return configs

    def compute_radius(pts):
        """Compute the valid radius for a configuration."""
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist, min_wall_dist)

    # Binary search for the best radius
    # Known best is ~0.11138. Let's search in [0.10, 0.12]
    lo = 0.10
    hi = 0.12
    
    best_radius = 0.0
    best_pts = None
    
    initial_configs = get_initial_configs()
    
    # We will do a few iterations of binary search
    # For each mid radius, we try all initial configs and keep the one with lowest energy
    # If min energy is close to 0, we accept the radius.
    
    for _ in range(8): # 8 binary search steps
        mid = (lo + hi) / 2.0
        
        best_energy_for_mid = np.inf
        best_pts_for_mid = None
        
        for config in initial_configs:
            pts_candidate, energy = optimize_for_radius(mid, config, time_limit=5.0)
            if energy < best_energy_for_mid:
                best_energy_for_mid = energy
                best_pts_for_mid = pts_candidate
        
        if best_energy_for_mid < 1e-4:
            # We found a valid config for this radius
            # Calculate the actual radius achieved to be precise
            r_achieved = compute_radius(best_pts_for_mid)
            if r_achieved > best_radius:
                best_radius = r_achieved
                best_pts = best_pts_for_mid
            lo = mid
        else:
            hi = mid
            
        # Check time
        if time.time() > 100:
            break
            
    # If we haven't found a good config, fallback to a simple grid
    if best_pts is None:
        ticks = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_radius = compute_radius(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
