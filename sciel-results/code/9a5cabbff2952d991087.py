import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    def compute_radius(pts):
        """Compute the valid radius for a configuration."""
        if len(pts) == 0:
            return 0.0
        # Pairwise distances
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0 if len(dists) > 0 else np.inf
        
        # Wall distances
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    def compute_energy(pts, r_target):
        """
        Compute energy for simulated annealing.
        Energy is high when constraints are violated.
        """
        n = len(pts)
        energy = 0.0
        
        # Pairwise constraint: dist(i,j) >= 2*r_target
        dists = pdist(pts)
        violations = 2.0 * r_target - dists
        energy += np.sum(np.maximum(violations, 0.0) ** 2)
        
        # Wall constraint: |x| <= 0.5 - r_target
        wall_violations = np.abs(pts) - (0.5 - r_target)
        energy += np.sum(np.maximum(wall_violations, 0.0) ** 2)
        
        return energy

    def simulated_annealing(pts_init, r_target, time_limit=30.0, seed=42):
        """
        Run simulated annealing to find a valid configuration.
        Returns (best_pts, best_radius) where best_radius >= r_target if found.
        """
        rng = np.random.default_rng(seed)
        start_time = time.time()
        
        pts = pts_init.copy()
        current_energy = compute_energy(pts, r_target)
        best_pts = pts.copy()
        best_energy = current_energy
        
        # Annealing parameters
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.995
        step_size = 0.02
        
        while (time.time() - start_time) < time_limit and T > T_min:
            # Generate a perturbation
            # Move one random point by a small amount
            idx = rng.integers(0, N)
            delta = rng.normal(0, step_size, size=3)
            new_pts = pts.copy()
            new_pts[idx] += delta
            new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_energy = compute_energy(new_pts, r_target)
            delta_E = new_energy - current_energy
            
            # Accept or reject
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                pts = new_pts
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            
            # Cool down
            T *= cooling_rate
            
            # Adapt step size
            if best_energy == 0:
                break
            if rng.random() < 0.01:
                step_size *= 0.95
                step_size = max(step_size, 1e-4)
        
        return best_pts, best_energy

    def binary_search_radius(pts_init, time_limit=30.0, seed=42):
        """
        Binary search for the maximum radius using simulated annealing.
        """
        lo = 0.05
        hi = 0.12
        
        # First, try to find a valid configuration at hi
        best_pts = pts_init.copy()
        best_radius = 0.0
        
        for _ in range(3):  # 3 iterations of binary search
            mid = (lo + hi) / 2.0
            pts_candidate, energy = simulated_annealing(pts_init, mid, time_limit=time_limit/3, seed=seed)
            
            if energy < 1e-6:
                # Valid configuration found at radius mid
                best_pts = pts_candidate
                best_radius = mid
                lo = mid
            else:
                hi = mid
        
        return best_pts, best_radius

    # Multiple initial configurations
    best_pts_overall = None
    best_radius_overall = -np.inf
    
    # Different grid structures
    configs = []
    
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.38, 0.38, nx)
    ys = np.linspace(-0.38, 0.38, ny)
    zs = np.linspace(-0.38, 0.38, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    configs.append(grid)
    
    # 4x5x5 grid
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.38, 0.38, nx)
    ys = np.linspace(-0.38, 0.38, ny)
    zs = np.linspace(-0.38, 0.38, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    configs.append(grid)
    
    # 5x4x5 grid
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.38, 0.38, nx)
    ys = np.linspace(-0.38, 0.38, ny)
    zs = np.linspace(-0.38, 0.38, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    configs.append(grid)
    
    # Random initialization
    rng = np.random.default_rng(123)
    random_pts = rng.uniform(-0.38, 0.38, size=(N, 3))
    configs.append(random_pts)
    
    # Body-centered cubic-like pattern
    # 5x5x4 with offsets
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.35, 0.35, nx)
    ys = np.linspace(-0.35, 0.35, ny)
    zs = np.linspace(-0.35, 0.35, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add body center offsets for some points
    if len(grid) == N:
        offsets = np.array([0.07, 0.07, 0.07])
        grid = grid + rng.uniform(-0.02, 0.02, size=(N, 3))
    configs.append(grid)
    
    start_total = time.time()
    time_per_config = 25.0  # Leave some buffer
    
    for i, config in enumerate(configs):
        if time.time() - start_total > 100.0:
            break
        
        seed = 42 + i
        pts_result, r_result = binary_search_radius(config, time_limit=time_per_config, seed=seed)
        
        if r_result > best_radius_overall:
            best_radius_overall = r_result
            best_pts_overall = pts_result
        
        # Early termination if we beat known best
        if best_radius_overall > 0.11138:
            break
    
    if best_pts_overall is None or best_radius_overall <= 0:
        # Fallback: simple grid
        ticks = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts_overall
# EVOLVE-BLOCK-END
