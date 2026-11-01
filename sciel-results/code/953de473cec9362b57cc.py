"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization
    # Use a 5x5x4 grid as a reasonable starting point, slightly perturbed to break symmetry
    # 100 points: 5x5x4 = 100
    # x: 5, y: 5, z: 4
    xs = np.linspace(-0.5, 0.5, 5, endpoint=False) + 0.1 # shift to center properly?
    # Let's use the standard grid spacing
    nx, ny, nz = 5, 5, 4
    # Spacing
    dx = 1.0 / nx
    dy = 1.0 / ny
    dz = 1.0 / nz
    
    # Centers
    x_coords = np.arange(nx) * dx - 0.5 + dx/2
    y_coords = np.arange(ny) * dy - 0.5 + dy/2
    z_coords = np.arange(nz) * dz - 0.5 + dz/2
    
    grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)

    def compute_radius(c):
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            r_pair = 0.0
        return min(r_wall, r_pair)

    best_centers = centers.copy()
    best_r = compute_radius(best_centers)
    
    # Force-directed refinement
    # We want to maximize the minimum distance.
    # We simulate forces: repulsive if too close.
    
    current_r = best_r
    base_step = 0.02
    decay = 0.9995
    stagnation = 0
    max_stagnation = 500
    
    it = 0
    while time.time() < deadline:
        it += 1
        
        # Compute current state
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        
        try:
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            min_pair_dist = np.min(dists[:, 1])
            r_pair = min_pair_dist / 2.0
        except Exception:
            break
            
        current_r = min(r_wall, r_pair)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                # If stuck, try a small random kick or reduce step
                if base_step > 0.001:
                    base_step *= 0.9
                else:
                    break

        # Determine target radius for forces
        # We push spheres apart to increase the minimum distance
        # Target slightly larger than current
        target_r = current_r + 0.001
        
        force = np.zeros_like(centers)
        
        # 1. Wall Forces
        # If a sphere is closer to a wall than target_r, push it away
        for k in range(3):
            d_wall = 0.5 - np.abs(centers[:, k])
            mask = d_wall < target_r
            if np.any(mask):
                viol = target_r - d_wall[mask]
                # Direction away from wall
                # If center[k] > 0, push in -x direction? No, away from wall.
                # Wall is at 0.5. Center is at c. Distance is 0.5 - c.
                # If 0.5 - c < target, c is too close to 0.5. Push towards 0 (negative direction).
                # If center[k] < 0, wall is at -0.5. Distance is c - (-0.5) = c + 0.5.
                # If c + 0.5 < target, c is too close to -0.5. Push towards 0 (positive direction).
                
                # Generalized:
                # Force direction should be towards the center of the cube (0) if near wall.
                # sign(centers[:, k]) points away from center.
                # So -sign(centers[:, k]) points towards center.
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                force[mask, k] += viol * dirs
        
        # 2. Pair Forces
        # If distance between two spheres < 2 * target_r, push apart
        target_dist = 2 * target_r
        try:
            pairs = tree.query_pairs(target_dist)
        except Exception:
            pairs = []
            
        for i, j in pairs:
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d > 1e-10 and d < target_dist:
                direction = diff / d
                viol = target_dist - d
                force[i] += viol * direction
                force[j] -= viol * direction
        
        # Apply forces
        norm_force = np.linalg.norm(force)
        if norm_force > 1e-10:
            # Adaptive step size
            step = base_step * decay ** it
            # Normalize force to avoid huge jumps
            force_norm = force / norm_force
            centers += step * force_norm
            centers = np.clip(centers, -0.499999, 0.499999)
            
        # Periodic check for improvement
        if it % 10 == 0:
             pass # compute_radius is done at start of loop

    return best_centers
# EVOLVE-BLOCK-END
