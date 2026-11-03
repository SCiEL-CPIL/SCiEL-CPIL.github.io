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
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def force_directed_relaxation(centers, iterations=5000, dt_initial=0.005, dt_final=0.0001, k_repulse=1.0, k_bound=10.0, cooling=0.9995):
        """
        Global force-directed relaxation using Lennard-Jones-like repulsion and boundary penalties.
        Moves all points simultaneously to maximize minimum distance.
        """
        centers = centers.copy()
        dt = dt_initial
        
        for it in range(iterations):
            if time.time() > deadline - 10:
                break
                
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion (soft Lennard-Jones style)
            # We want to push points apart when they are close.
            # Force magnitude ~ 1/d^2 for close points, but we want to maximize min distance.
            # A better proxy: Force = (1/d^2) * (direction) for all pairs, weighted by how close they are to the current min?
            # Actually, standard FDR for packing uses repulsion for all pairs.
            # To be efficient, we can limit to neighbors, but N=200 is small enough for O(N^2) if optimized.
            # cdist is fast.
            
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Avoid division by zero
            dist_matrix_safe = np.where(dist_matrix < 1e-10, 1e-10, dist_matrix)
            
            # Force vector: repulsion
            # F_ij = (1 / d_ij^2) * (c_i - c_j) / d_ij  => (c_i - c_j) / d_ij^3
            # This is strong at short range.
            
            # Normalize direction
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (N, N, 2)
            # dist_matrix is (N, N)
            # Force on i from j: diff[i,j] / dist^3
            # We sum over j
            
            # To avoid memory explosion for N=200, N^2*2 is 80,000 floats, which is fine.
            inv_dist_cubed = 1.0 / (dist_matrix_safe ** 3)
            # Mask diagonal
            np.fill_diagonal(inv_dist_cubed, 0.0)
            
            # Force = sum_j ( (c_i - c_j) / d_ij^3 )
            # diff is (N, N, 2). inv_dist_cubed is (N, N).
            # Multiply diff by inv_dist_cubed[..., np.newaxis]
            repulsion_forces = np.sum(diff * inv_dist_cubed[..., np.newaxis], axis=1)
            
            # Boundary forces
            # Push points away from boundaries if they are too close.
            # Boundary distance d_bound = 0.5 - |c|
            # If d_bound < r_target (or some threshold), apply force towards center.
            # Let's use a penalty force that increases as we approach the boundary.
            # Force direction: towards center (sign of c is opposite to force)
            # Magnitude: k_bound / d_bound^2 ? Or just linear?
            # Let's use a soft penalty: if d_bound < 0.05, push in.
            # Actually, to maximize radius, we want points to be at least r away.
            # The optimal configuration will have some points touching the boundary.
            # So we shouldn't push ALL points away from boundary strongly, only those that are "stuck" or violating?
            # In FDR, usually we have a hard constraint or a very strong penalty.
            # Let's use a penalty force that is zero if d_bound > 0.02, and increases as d_bound decreases.
            # F_bound = -k_bound * (0.02 - d_bound) * sign(c) if d_bound < 0.02 else 0
            # This keeps points away from the very edge, allowing them to settle at optimal r.
            
            d_bound_x = 0.5 - np.abs(centers[:, 0])
            d_bound_y = 0.5 - np.abs(centers[:, 1])
            
            threshold = 0.02
            penalty_x = np.where(d_bound_x < threshold, k_bound * (threshold - d_bound_x) ** 2, 0.0)
            penalty_y = np.where(d_bound_y < threshold, k_bound * (threshold - d_bound_y) ** 2, 0.0)
            
            # Force direction is towards center: -sign(c)
            sign_x = np.sign(centers[:, 0])
            sign_y = np.sign(centers[:, 1])
            # If c is 0, sign is 0, force is 0. This is fine.
            
            boundary_forces = np.column_stack([-sign_x * penalty_x, -sign_y * penalty_y])
            
            total_forces = repulsion_forces + boundary_forces
            
            # Normalize forces to prevent explosion
            force_norms = np.linalg.norm(total_forces, axis=1)
            max_force = 0.1
            scale = np.where(force_norms > max_force, max_force / force_norms, 1.0)
            total_forces *= scale[:, np.newaxis]
            
            # Update positions
            centers += dt * total_forces
            
            # Enforce hard boundaries (clip)
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            # Cool down
            dt *= cooling
            
            # Optional: Local optimization every 100 steps? No, let FDR do its thing.
            
        return centers

    def local_polish(centers, iterations=100):
        """
        Fine-tune using the bottleneck method to squeeze out last bits of radius.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # Find bottleneck
            wall_dists = 0.5 - np.abs(centers)
            min_wall_idx = np.argmin(wall_dists)
            min_wall = wall_dists.flat[min_wall_idx]
            
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            min_pair_idx = np.argmin(dist_matrix)
            min_pair = dist_matrix.flat[min_pair_idx]
            
            if min_wall < min_pair / 2.0:
                # Wall bottleneck
                i = min_wall_idx // 2
                orig = centers[i].copy()
                # Move away from wall
                if centers[i, 0] < 0:
                    direction = np.array([1, 0])
                elif centers[i, 0] > 0:
                    direction = np.array([-1, 0])
                elif centers[i, 1] < 0:
                    direction = np.array([0, 1])
                else:
                    direction = np.array([0, -1])
                
                best_pos = orig
                best_r = current_r
                for step in [0.0001, 0.0005, 0.001]:
                    new_center = orig + direction * step
                    if -0.5 <= new_center[0] <= 0.5 and -0.5 <= new_center[1] <= 0.5:
                        centers[i] = new_center
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos = new_center.copy()
                centers[i] = best_pos
            else:
                # Pair bottleneck
                i, j = np.unravel_index(min_pair_idx, (N, N))
                orig_i = centers[i].copy()
                orig_j = centers[j].copy()
                direction = (orig_i - orig_j)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                else:
                    direction = np.array([1, 0])
                
                best_pos_i = orig_i
                best_pos_j = orig_j
                best_r = current_r
                
                for step in [0.0001, 0.0005, 0.001]:
                    # Move i away
                    new_i = orig_i + direction * step
                    if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5:
                        centers[i] = new_i
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos_i = new_i.copy()
                    centers[i] = orig_i
                    
                    # Move j away
                    new_j = orig_j - direction * step
                    if -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
                        centers[j] = new_j
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos_j = new_j.copy()
                    centers[j] = orig_j
                
                centers[i] = best_pos_i
                centers[j] = best_pos_j
            
            new_r = compute_radius(centers)
            if abs(new_r - current_r) < 1e-14:
                # Perturb random point to escape local minimum
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.001
                centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                continue
            current_r = new_r
            
        return centers, compute_radius(centers)

    def hexagonal_init():
        """Generate a hexagonal lattice initial guess."""
        # Approximate hexagonal packing
        # r ~ 0.0366, d ~ 0.073
        # Rows: ~ 1 / (d * sqrt(3)/2) ~ 1 / 0.063 ~ 16 rows
        # Cols: ~ 1 / d ~ 14 cols
        # Total ~ 224 points. We need 200.
        
        rows = 14
        cols = 15
        d = 1.0 / cols
        y_spacing = d * math.sqrt(3) / 2
        
        pts = []
        total_height = (rows - 1) * y_spacing
        y_start = -total_height / 2
        
        for i in range(rows):
            y = y_start + i * y_spacing
            if i % 2 == 0:
                x_start = -0.5 + d/2
                for j in range(cols):
                    x = x_start + j * d
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for j in range(cols - 1):
                    x = x_start + j * d
                    pts.append([x, y])
        
        pts = np.array(pts)
        # If we have more than N, randomly sample N
        if len(pts) > N:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            # Pad with random points
            diff = N - len(pts)
            random_pts = np.random.uniform(-0.5, 0.5, (diff, 2))
            pts = np.vstack([pts, random_pts])
            
        return pts

    def square_init():
        """Generate a square grid initial guess."""
        m = int(math.ceil(math.sqrt(N)))
        # Use m x m grid, take first N
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) > N:
            idx = np.random.choice(len(grid), N, replace=False)
            grid = grid[idx]
        return grid

    def random_init():
        """Generate random initial guess."""
        return np.random.uniform(-0.5, 0.5, (N, 2))

    best_centers = None
    best_radius = 0.0
    
    # Strategy: Multi-restart with FDR
    # 1. Hexagonal init
    # 2. Square init
    # 3. Random inits
    
    inits = [
        ("hex", hexagonal_init()),
        ("sq", square_init()),
    ]
    
    # Add a few random inits if time permits
    # Estimate time per FDR run: 5000 iters * O(N^2) ~ 5000 * 40000 ops = 200M ops.
    # NumPy cdist is fast, maybe 1-2 seconds per run?
    # Let's run 2 FDR runs per init.
    
    for name, init_centers in inits:
        if time.time() > deadline - 20:
            break
            
        # Run FDR
        relaxed = force_directed_relaxation(init_centers, iterations=3000, dt_initial=0.002, dt_final=0.0001, k_repulse=1.0, k_bound=5.0, cooling=0.999)
        
        # Polish
        polished, r = local_polish(relaxed, iterations=50)
        
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
    # If time remains, try random inits
    if time.time() < deadline - 30:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            init_centers = random_init()
            relaxed = force_directed_relaxation(init_centers, iterations=2000, dt_initial=0.001, dt_final=0.0001, k_repulse=1.0, k_bound=5.0, cooling=0.999)
            polished, r = local_polish(relaxed, iterations=50)
            
            if r > best_radius:
                best_radius = r
                best_centers = polished.copy()

    # Final check
    if best_centers is None:
        best_centers = square_init()
        
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
