import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    best_r = 0.0
    best_centers = None
    
    def get_min_radius(centers):
        # Compute pairwise distances
        # Using cdist is often faster and more stable than manual broadcasting for N=100
        dist = cdist(centers, centers)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distances
        # Distance from center to closest wall
        # Walls are at +/- 0.5
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def optimize(centers, max_steps=20000, initial_step=0.01, beta=10.0):
        """
        Force-directed relaxation.
        Forces:
        1. Pairwise repulsion: ~ 1/d^2
        2. Wall repulsion: ~ 1/d_wall^2 (pushing away from walls)
        
        We maximize the minimum distance, which is equivalent to minimizing the energy
        where energy is dominated by the closest pairs/walls.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = initial_step
        no_improve_count = 0
        max_no_improve = 500
        
        # Pre-allocate for forces
        forces = np.zeros_like(centers)
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # 1. Compute Pairwise Repulsion
            # dist[i,j] is distance between i and j
            dist = cdist(centers, centers)
            # Avoid division by zero
            dist[dist == 0] = 1e-6
            
            # We want to push apart points that are close.
            # Force magnitude proportional to 1/d^2
            # To avoid huge forces from very close points, we cap the force or use a softening parameter
            # Let's use a simple inverse square law with a cap
            inv_dist_sq = 1.0 / (dist**2)
            np.fill_diagonal(inv_dist_sq, 0.0)
            
            # Cap the force to prevent instability
            max_force_val = 100.0
            inv_dist_sq = np.minimum(inv_dist_sq, max_force_val)
            
            # forces[i] = sum_j (centers[i] - centers[j]) / dist[i,j]^2
            # Vectorized calculation:
            # diff[i,j] = centers[i] - centers[j]
            # force[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
            
            # This is a matrix multiplication trick:
            # sum_j (centers[i] - centers[j]) * w[i,j]
            # = centers[i] * sum_j w[i,j] - sum_j centers[j] * w[i,j]
            
            row_sums = np.sum(inv_dist_sq, axis=1)
            weighted_centers = inv_dist_sq @ centers
            
            pair_forces = centers * row_sums[:, np.newaxis] - weighted_centers
            
            # 2. Compute Wall Repulsion
            # We want to push spheres away from the walls.
            # If a sphere is close to the +x wall (x ~ 0.5), push it in -x direction.
            # If a sphere is close to the -x wall (x ~ -0.5), push it in +x direction.
            
            # Distance to walls for each coordinate
            # dist_to_pos = 0.5 - c
            # dist_to_neg = 0.5 + c
            
            # Force from +x wall: -sign * strength / dist^2
            # If c > 0, it's closer to +x wall? No, distance to +x is 0.5-c.
            # Let's define wall_force_k for coordinate k.
            
            wall_forces = np.zeros_like(centers)
            
            for k in range(3):
                c_k = centers[:, k]
                d_pos = 0.5 - c_k  # Distance to +0.5 wall
                d_neg = 0.5 + c_k  # Distance to -0.5 wall
                
                # We only care if they are close. 
                # Let's say if dist < 0.2, apply force.
                # Force magnitude: strength / d^2
                # Direction: Away from wall.
                
                # From +x wall: force is -x direction. Magnitude ~ 1/d_pos^2
                # From -x wall: force is +x direction. Magnitude ~ 1/d_neg^2
                
                # To avoid numerical issues with very small d, we can add a small epsilon or just cap.
                # Let's use a soft potential: F = -d / (d^2 + eps)^2 ? 
                # Simpler: F = 1 / (d^2 + eps)
                
                eps = 1e-4
                f_pos = 1.0 / (d_pos**2 + eps)
                f_neg = 1.0 / (d_neg**2 + eps)
                
                # Cap forces
                max_wall_force = 100.0
                f_pos = np.minimum(f_pos, max_wall_force)
                f_neg = np.minimum(f_neg, max_wall_force)
                
                # Net force in k direction:
                # Push away from +x (negative direction): -f_pos
                # Push away from -x (positive direction): +f_neg
                wall_forces[:, k] = f_neg - f_pos
            
            # Total Force
            forces = pair_forces + wall_forces
            
            # Normalize forces to prevent large steps
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes)
            if max_force > 0:
                forces /= max_force
            
            # Update positions
            centers += step * forces
            
            # Clamp to box (with small margin to avoid being exactly on boundary which causes 0 radius)
            # But we want them as close to boundary as possible.
            # Clamping to [-0.5, 0.5] is safe. The radius calculation handles the distance.
            # If a point goes outside, we clamp it back.
            centers = np.clip(centers, -0.5, 0.5)
            
            # Compute new radius
            new_r = get_min_radius(centers)
            
            # Adaptive step size
            if new_r > current_r:
                step = min(step * 1.05, 0.05)
                no_improve_count = 0
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                step = max(step * 0.95, 0.0001)
                no_improve_count += 1
            
            current_r = new_r
            
            # If stuck, add random perturbation
            if no_improve_count > max_no_improve:
                n_perturb = max(1, N // 10)
                perturbed = np.random.choice(N, n_perturb, replace=False)
                centers[perturbed] += np.random.randn(n_perturb, 3) * 0.01
                centers = np.clip(centers, -0.5, 0.5)
                no_improve_count = 0
                step = 0.005
        
        return best_local_centers, best_local_r

    # Initial Configurations
    
    # 1. Cubic Grid (Seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    c1, r1 = optimize(init_grid, max_steps=15000, initial_step=0.01)
    if r1 > best_r:
        best_r = r1
        best_centers = c1

    # 2. Random Initialization
    np.random.seed(42)
    init_random = np.random.uniform(-0.45, 0.45, (N, 3))
    c2, r2 = optimize(init_random, max_steps=15000, initial_step=0.01)
    if r2 > best_r:
        best_r = r2
        best_centers = c2

    # 3. FCC-like Initialization
    # Generate a dense FCC lattice and pick the best N
    s = 0.2
    pts = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                base = np.array([i, j, k]) * s
                pts.append(base)
                pts.append(base + np.array([s/2, s/2, 0]))
                pts.append(base + np.array([s/2, 0, s/2]))
                pts.append(base + np.array([0, s/2, s/2]))
    
    fcc_grid = np.array(pts)
    fcc_grid -= fcc_grid.mean(axis=0)
    extent = np.max(np.abs(fcc_grid))
    if extent > 0.5:
        fcc_grid *= (0.5 / extent)
    
    # Pick N points that are most "central" or just random slice?
    # Let's sort by distance from center and take the innermost N?
    # Or just take a random subset?
    # Taking innermost might cluster them too much.
    # Let's take a random subset of the FCC grid to ensure spread.
    idx = np.random.choice(len(fcc_grid), N, replace=False)
    init_fcc = fcc_grid[idx]
    
    c3, r3 = optimize(init_fcc, max_steps=15000, initial_step=0.01)
    if r3 > best_r:
        best_r = r3
        best_centers = c3

    # 4. Final Polish on the best configuration
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        # Use smaller steps for high precision
        c_best, r_best = optimize(best_centers, max_steps=10000, initial_step=0.001)
        if r_best > best_r:
            best_r = r_best
            best_centers = c_best

    if best_centers is None:
        return init_grid

    return best_centers
