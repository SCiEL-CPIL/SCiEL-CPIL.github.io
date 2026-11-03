"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        # Check bounds
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Compute pairwise distances
        n = len(centers)
        if n < 2:
            return max(0.0, min_wall)
            
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def initialize_hex_grid():
        """
        Initialize centers on a hexagonal grid that fits inside the square.
        We aim for roughly 14x14 = 196 points, then add a few more or adjust.
        Actually, let's generate a dense hex grid and pick the best N.
        """
        # Try to fit a hex grid
        # Hex grid with spacing d.
        # Rows: y = i * d * sqrt(3)/2
        # Cols: x = j * d + (i % 2) * d/2
        
        # We want to maximize the minimum distance.
        # Let's try a few configurations of rows/cols.
        
        best_centers = None
        best_r = 0.0
        
        # Scan possible number of rows
        for n_rows in range(12, 16):
            # Calculate max cols that fit in width 1.0
            # Width = (n_cols - 1) * d + d/2 (for odd rows) or (n_cols-1)*d (for even)
            # Let's just generate a grid with a specific d and see how many fit.
            
            # Estimate d from area: Area = 1.0. N=200.
            # Hex packing density ~ 0.9069.
            # Area per circle ~ pi * r^2. 
            # Total area ~ N * pi * r^2 / 0.9069 <= 1.
            # r ~ sqrt(1 / (N * pi / 0.9069)) ~ sqrt(0.9069 / (200 * 3.14)) ~ sqrt(0.9069/628) ~ sqrt(0.00144) ~ 0.038
            
            # Let's try d values around 0.07 - 0.08 (since d = 2r ~ 0.076)
            
            for d in np.linspace(0.06, 0.09, 10):
                pts = []
                row = 0
                y = -0.5 + d/2  # Start with margin
                while y <= 0.5 - d/2:
                    x = -0.5 + d/2
                    if row % 2 == 1:
                        x = -0.5 + d  # Offset by d (or d/2? standard hex is d/2 offset in x, but let's stick to simple grid first)
                        # Standard hex: x_offset = d/2 for odd rows
                        x = -0.5 + d/2 + d/2 # = -0.5 + d ? No.
                        # Let's do:
                        # Row 0: x = -0.5 + d/2, -0.5 + 3d/2, ...
                        # Row 1: x = -0.5 + d, -0.5 + 2d, ...
                        pass
                    
                    # Let's use a standard hex grid generator
                    x_offset = 0.0 if row % 2 == 0 else d / 2.0
                    x = -0.5 + d / 2.0 + x_offset
                    while x <= 0.5 - d / 2.0:
                        pts.append([x, y])
                        x += d
                    y += d * math.sqrt(3) / 2.0
                    row += 1
                
                pts = np.array(pts)
                if len(pts) < N:
                    continue
                
                # If we have more than N, we need to select N.
                # For now, just take the first N? No, that biases to top-left.
                # Better: take the ones closest to the center? Or just random?
                # Actually, if we generate a grid that fits exactly N, that's ideal.
                # But it's hard to hit exactly N.
                # Let's just take the first N and rely on MD to fix it.
                # Or better, take the N points that are most "central" to avoid edge bias?
                # Let's just take the first N for now and let MD handle it.
                
                # To be more robust, let's try to center the grid.
                # Calculate bounding box of pts
                min_x, min_y = np.min(pts, axis=0)
                max_x, max_y = np.max(pts, axis=0)
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                
                # Shift to center at (0,0)
                pts_centered = pts - np.array([center_x, center_y])
                
                # If we have too many, we might need to trim.
                # If we have too few, we can't use this d.
                
                if len(pts_centered) >= N:
                    # Select N points. To maximize radius, we want them spread out.
                    # Taking the first N is bad if the grid is rectangular.
                    # Let's take a random subset? Or the ones with max distance to boundary?
                    # For initialization, a random subset of the grid is fine.
                    idx = np.random.choice(len(pts_centered), N, replace=False)
                    init_pts = pts_centered[idx]
                    
                    r = compute_radius(init_pts)
                    if r > best_r:
                        best_r = r
                        best_centers = init_pts.copy()
        
        if best_centers is None:
            # Fallback: Square grid
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_centers = grid[:N]
            
        return best_centers

    def force_directed_relaxation(centers, max_time=120.0):
        """
        Optimize centers using force-directed relaxation (Molecular Dynamics).
        Forces:
        1. Repulsion between circles: F = k * (1/d^2) * (r_i - r_j) / d ... 
           Actually, simpler: F = k * (1/d^2) * direction.
           We want to maximize min distance.
           Let's use a potential energy U = sum_{i<j} 1/d_ij^2 + sum_i 1/wall_dist_i^2.
           Gradient descent on U.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        # Parameters
        dt = 0.001  # Time step
        k_rep = 1.0  # Repulsion strength
        k_wall = 10.0  # Wall repulsion strength (stronger to keep inside)
        
        # Cooling schedule
        temp = 1.0
        cooling_rate = 0.999
        
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            if temp < 1e-6:
                break
                
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            # Use cdist for efficiency
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Avoid division by zero
            dist_matrix_safe = np.maximum(dist_matrix, 1e-6)
            
            # Force magnitude: 1/d^2
            force_mag = 1.0 / (dist_matrix_safe ** 2)
            
            # Direction: (c_i - c_j) / d
            # F_ij on i is in direction (c_i - c_j)
            # F_ij on j is in direction (c_j - c_i) = - (c_i - c_j)
            
            # Vectorized calculation of forces
            # diff[i,j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            
            # Normalize diff to get direction
            norms = dist_matrix_safe[..., np.newaxis]
            dir_vec = diff / norms
            
            # F_i = sum_j force_mag[i,j] * dir_vec[i,j]
            forces += np.sum(force_mag[..., np.newaxis] * dir_vec, axis=1)
            
            # Wall repulsion
            # Push away from walls if close
            # Wall distance: 0.5 - |x|
            # Force direction: sign(x) if x>0, -sign(x) if x<0? 
            # Actually, force should push towards center.
            # If x > 0, force is -x direction (towards 0).
            # If x < 0, force is +x direction (towards 0).
            # So force_x = -k_wall * x / (wall_dist^2) ?
            
            # Let's use a smooth potential for walls.
            # U_wall = 1 / (0.5 - |x|)^2
            # dU/dx = 2 * (0.5 - |x|)^-3 * d(0.5-|x|)/dx
            # d(0.5-|x|)/dx = -sign(x)
            # So F_x = -dU/dx = 2 * sign(x) / (0.5 - |x|)^3
            
            wall_dist_x = 0.5 - np.abs(centers[:, 0])
            wall_dist_y = 0.5 - np.abs(centers[:, 1])
            
            # Clip to avoid singularity
            wall_dist_x_safe = np.maximum(wall_dist_x, 1e-6)
            wall_dist_y_safe = np.maximum(wall_dist_y, 1e-6)
            
            sign_x = np.sign(centers[:, 0])
            sign_y = np.sign(centers[:, 1])
            
            # If center is exactly 0, sign is 0, force is 0. That's fine.
            # But if it's very close to 0, it's stable.
            
            forces[:, 0] += k_wall * sign_x / (wall_dist_x_safe ** 3)
            forces[:, 1] += k_wall * sign_y / (wall_dist_y_safe ** 3)
            
            # Update positions
            # Velocity Verlet or simple Euler?
            # Simple Euler with damping.
            # v = v * damping + F * dt
            # We don't store velocity, so just use force as velocity proxy?
            # Let's use a simple gradient step.
            
            # Normalize forces to prevent explosion
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force > 1e6:
                forces /= max_force * 1e6
            
            centers += forces * dt * temp
            
            # Project back to square if outside
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            # Compute radius
            new_r = compute_radius(centers)
            
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
            else:
                # If it got worse, revert? 
                # In MD, we usually don't revert, but for optimization, 
                # we can accept worse moves with probability exp(-delta/E).
                # Let's keep it simple: always accept, but cool down.
                # If it gets significantly worse, we might be stuck.
                # Let's add a small random jitter to escape local minima.
                if new_r < best_r - 1e-4:
                    # Add jitter
                    jitter = np.random.randn(*centers.shape) * 0.001 * temp
                    centers += jitter
                    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
                    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            current_r = new_r
            temp *= cooling_rate
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        return best_centers, best_r

    # 1. Initialize
    centers = initialize_hex_grid()
    
    # 2. Relax
    refined_centers, refined_r = force_directed_relaxation(centers, max_time=130.0)
    
    # 3. Final check and return
    final_r = compute_radius(refined_centers)
    
    # Ensure validity
    if not np.all(np.isfinite(refined_centers)):
        # Fallback
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        refined_centers = grid[:N]
        
    return refined_centers

# EVOLVE-BLOCK-END
