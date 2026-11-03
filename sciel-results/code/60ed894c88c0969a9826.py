import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def square_grid_init(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def md_optimize(centers, r, time_limit=5.0):
        """
        Molecular Dynamics optimization with hard constraints.
        Centers are kept inside [-0.5+r, 0.5-r]^2.
        Circles are pushed apart if distance < 2r.
        Returns final centers.
        """
        centers = centers.copy()
        # Initial projection to valid box
        box_min = -0.5 + r
        box_max = 0.5 - r
        centers = np.clip(centers, box_min, box_max)
        
        # If r is too large for the box, return as is (invalid packing will be caught by score)
        if box_min >= box_max:
            return centers

        # Initialize velocities
        v = np.random.randn(N, 2) * 0.01
        # v = np.zeros((N, 2)) # Zero init is more stable for hard constraints
        
        # Parameters
        dt = 0.005
        damping = 0.98
        push_strength = 1.0
        
        start = time.time()
        
        # Precompute indices for pairs? No, N=200 is small enough for O(N^2) per step if optimized.
        # But we can use cdist or broadcasting.
        
        while time.time() < start + time_limit:
            # Compute forces
            # Diff[i,j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0) # Avoid div by 0
            
            # Force is zero if dist > 2r
            # If dist <= 2r, push apart
            mask = dists < 2.0 * r
            # Exclude diagonal
            mask = np.where(np.eye(N, dtype=bool), False, mask)
            
            forces = np.zeros_like(centers)
            if np.any(mask):
                # Overlap amount
                overlap = 2.0 * r - dists
                # Direction
                # force on i due to j is +diff[i,j]/dist * magnitude
                # magnitude = overlap
                # factor = overlap / dist
                # Only where mask is true
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor = np.where(mask, overlap / dists, 0.0)
                
                # F_i = sum_j factor[i,j] * diff[i,j]
                forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            # Wall forces
            # If center is close to wall, push inwards
            # Box is [box_min, box_max]
            # Wall distance:
            # left wall: x - box_min. If < 0 (should not happen due to clip) or small, push right.
            # Actually, we clip after update, so we just need to handle the "soft" wall or just clip.
            # For MD, hard clipping is often sufficient if dt is small.
            # But to be safe, let's add a wall force.
            # d_left = centers[:, 0] - box_min
            # d_right = box_max - centers[:, 0]
            # If d_left < margin, force_x += k * (margin - d_left)
            # Let's just use clipping for walls to save compute and ensure validity.
            
            # Update velocities
            v += forces * dt
            v *= damping
            
            # Update positions
            centers += v * dt
            
            # Clip to box
            centers = np.clip(centers, box_min, box_max)
            
            # Adaptive dt: if energy is high (many overlaps), reduce dt
            # Simple heuristic: if max overlap is large, reduce dt
            max_overlap = np.max(2.0 * r - dists[mask]) if np.any(mask) else 0.0
            if max_overlap > 0.01 * r:
                dt *= 0.9
            elif dt < 0.001:
                dt = 0.001
            elif max_overlap < 0.001 * r:
                dt = min(dt * 1.05, 0.005)

        return centers

    def binary_search_radius(init_centers, r_low, r_high, time_per_trial=3.0):
        """
        Find max r such that MD can find a valid packing.
        """
        best_valid_r = r_low
        best_centers = init_centers.copy()
        
        # We want to maximize r.
        # Start with a range.
        # The best known is ~0.0366.
        # Let's try to find if we can go higher.
        
        # To save time, we do a few binary search steps.
        # But MD is stochastic. A "valid" packing at r doesn't guarantee valid at r+eps.
        # However, if we find a valid packing at r, we can try to push it to r+delta.
        
        # Strategy:
        # 1. Run MD at current best r.
        # 2. If valid, try r * 1.01.
        # 3. If invalid, try r * 0.99.
        
        current_r = r_low
        # Ensure initial centers are valid for r_low? 
        # If not, we might fail immediately.
        # Let's just run MD.
        
        while time.time() < start_time + deadline - 5.0:
            # Run MD
            trial_centers = md_optimize(init_centers, current_r, time_limit=time_per_trial)
            
            # Check validity
            r_achieved = compute_radius(trial_centers)
            
            if r_achieved >= current_r - 1e-9:
                # Success
                best_valid_r = current_r
                best_centers = trial_centers
                # Increase r
                current_r *= 1.005
            else:
                # Failure
                # Decrease r
                current_r *= 0.99
                # If we drop too much, stop?
                if current_r < r_low * 0.9:
                    break
            
            # Safety check
            if time.time() > start_time + deadline - 10.0:
                break

        return best_valid_r, best_centers

    # Main Logic
    best_centers = None
    best_radius = 0.0

    # 1. Generate Initial Configurations
    inits = []
    
    # Hexagonal
    for n_rows in range(13, 17):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
    
    # Square Grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)
    
    # Random
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)

    # 2. Optimize each init
    # Start with a radius estimate. 
    # Hexagonal packing of N circles in unit square.
    # Approx r ~ 1 / (2 * sqrt(N)) * some_factor.
    # For N=200, sqrt(200) ~ 14.1. 1/28 ~ 0.035.
    # Let's start slightly lower to ensure validity, then grow.
    r_start = 0.03
    
    for init in inits:
        if time.time() > deadline - 15:
            break
            
        # Quick eval
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
            
        # Run Binary Search / MD optimization
        # We pass the init centers. MD will project them.
        # If init is already valid for r_start, MD will just relax it.
        # If not, MD will push them apart, potentially reducing r achieved.
        
        # Let's try to optimize from r_start.
        # If r_start is too high for this init, MD will fail to find valid config.
        # In that case, binary search will lower r.
        
        r_opt, c_opt = binary_search_radius(init, r_start, 0.04, time_per_trial=3.0)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt
        
        # Update r_start for next init? 
        # If we found a better radius, start next one from there?
        # No, different inits have different local minima.
        # But we can use the best found so far as a "warm start" for MD?
        # MD is sensitive to initial conditions.
        
    # 3. Final Polish
    # If we have a good candidate, we can try to do a very fine MD run with small dt
    # to squeeze out a bit more radius.
    if best_centers is not None and time.time() < deadline - 2:
        # Try to increase radius slightly
        r_current = best_radius
        # Try r_current * 1.001
        trial_r = r_current * 1.001
        trial_centers = md_optimize(best_centers, trial_r, time_limit=2.0)
        r_trial = compute_radius(trial_centers)
        if r_trial > best_radius:
            best_radius = r_trial
            best_centers = trial_centers

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
