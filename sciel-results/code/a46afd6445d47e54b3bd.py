import numpy as np
from scipy.spatial.distance import pdist
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    def compute_violations(pts, r_target):
        """
        Compute the total violation energy.
        Violations:
        1. Pairwise distance < 2 * r_target
        2. Wall distance < r_target
        """
        # Pairwise
        dists = pdist(pts)
        min_dist_required = 2.0 * r_target
        pair_violations = np.minimum(min_dist_required - dists, 0.0)
        # We want to penalize overlap. 
        # If dist < min_dist, violation is (min_dist - dist).
        # Let's use squared violations for smoothness.
        pair_violations = np.maximum(min_dist_required - dists, 0.0)
        pair_energy = np.sum(pair_violations ** 2)
        
        # Walls
        # Distance to wall is 0.5 - |coord|
        # Violation if r_target > 0.5 - |coord| => |coord| > 0.5 - r_target
        wall_violations = np.abs(pts) - (0.5 - r_target)
        wall_violations = np.maximum(wall_violations, 0.0)
        wall_energy = np.sum(wall_violations ** 2)
        
        return pair_energy + wall_energy

    def simulated_annealing(r_target, time_limit=5.0):
        """
        Run Simulated Annealing to minimize violations for a given r_target.
        """
        start_time = time.time()
        
        # Initialize with a jittered grid to start somewhat packed
        # 5x5x4 grid
        nx, ny, nz = 5, 5, 4
        xs = np.linspace(-0.45, 0.45, nx)
        ys = np.linspace(-0.45, 0.45, ny)
        zs = np.linspace(-0.45, 0.45, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        
        # Add small random jitter
        rng = np.random.default_rng()
        pts = grid + rng.uniform(-0.05, 0.05, size=(N, 3))
        
        # Clip to box
        pts = np.clip(pts, -0.5, 0.5)
        
        current_energy = compute_violations(pts, r_target)
        best_pts = pts.copy()
        best_energy = current_energy
        
        # SA Parameters
        T = 0.01 # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.9995
        step_size = 0.05
        
        while time.time() - start_time < time_limit:
            # Generate new solution
            idx = rng.integers(N)
            delta = rng.normal(0, step_size, size=3)
            
            new_pts = pts.copy()
            new_pts[idx] = np.clip(new_pts[idx] + delta, -0.5, 0.5)
            
            new_energy = compute_violations(new_pts, r_target)
            
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
                pts = new_pts
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            
            # Cool down
            T *= cooling_rate
            
            # Adaptive step size: if stuck, increase step; if improving, decrease?
            # Simple heuristic: decrease step size slowly
            step_size *= 0.9999
            
        return best_pts, best_energy

    def compute_radius(pts):
        """Compute the valid radius for a configuration."""
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist, min_wall_dist)

    # Binary search for the best radius
    lo = 0.10
    hi = 0.115 # Slightly above known best 0.11138
    
    best_radius = 0.0
    best_pts = None
    
    total_time_limit = 110.0 # Leave 10s margin
    start_total = time.time()
    
    # Perform binary search
    # We do a coarse search first, then fine-tune
    for i in range(10):
        if time.time() - start_total > total_time_limit:
            break
            
        mid = (lo + hi) / 2.0
        
        # Run SA for this radius
        # Give more time for higher radii (harder to pack)
        sa_time = 5.0 if i < 5 else 10.0
        
        pts_candidate, energy = simulated_annealing(mid, time_limit=sa_time)
        
        if energy < 1e-4:
            # Valid configuration found
            r_achieved = compute_radius(pts_candidate)
            if r_achieved > best_radius:
                best_radius = r_achieved
                best_pts = pts_candidate
            lo = mid
        else:
            hi = mid
            
    # If we haven't found a good config, fallback to a simple grid
    if best_pts is None:
        ticks = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_radius = compute_radius(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
