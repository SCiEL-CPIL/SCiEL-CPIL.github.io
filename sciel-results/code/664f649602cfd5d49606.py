import numpy as np
import time

N = 26
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def construct():
    best_score = 0
    best_circles = None
    
    # Helper to check time
    def time_up():
        return (time.time() - START_TIME) > TIME_LIMIT

    # Helper to refine radii for given centers
    def refine_radii(centers):
        """
        Given centers (n, 2), find max radii such that circles are disjoint and in box.
        Returns array of radii or None if invalid (overlap).
        """
        n = len(centers)
        if n == 0:
            return np.array([])
        
        # Initial radii: limited by box
        radii = np.min(np.stack([
            centers[:, 0], 
            1.0 - centers[:, 0], 
            centers[:, 1], 
            1.0 - centers[:, 1]
        ], axis=1), axis=1)
        
        # Iterative relaxation
        for _ in range(20):
            new_radii = radii.copy()
            for i in range(n):
                r_max = radii[i] # Start with current radius, can only increase? No, we want max possible.
                # Actually, we should recalculate from scratch based on others?
                # No, the constraint is r_i + r_j <= dist_ij.
                # We want to maximize sum r_i.
                # This is a linear programming problem, but we can use a simple iterative update.
                # r_i = min( box_limit_i, min_{j!=i} (dist_ij - r_j) )
                
                limit = np.min(np.stack([
                    centers[i, 0], 
                    1.0 - centers[i, 0], 
                    centers[i, 1], 
                    1.0 - centers[i, 1]
                ]))
                
                for j in range(n):
                    if i == j:
                        continue
                    dist = np.linalg.norm(centers[i] - centers[j])
                    # If dist - r_j is negative, overlap is unavoidable with current r_j
                    # But we are solving for r_i given r_j.
                    limit = min(limit, dist - radii[j])
                
                if limit < 0:
                    # This center is trapped by others
                    # In a valid packing, this shouldn't happen if we started valid.
                    # But if we just picked random centers, it might.
                    # We return None if any radius must be <= 0
                    return None
                
                new_radii[i] = limit
            
            # Check for convergence
            if np.allclose(radii, new_radii):
                radii = new_radii
                break
            radii = new_radii
        
        # Final check for validity
        if np.any(radii <= 0):
            return None
            
        # Verify disjoint interiors
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return None
        return radii

    def greedy_pack(grid_size, jitter_scale, seed):
        """
        Generate a packing using grid + jitter + greedy selection + refinement.
        """
        rng = np.random.default_rng(seed)
        
        # Generate grid
        xs = np.linspace(0.1, 0.9, grid_size)
        ys = np.linspace(0.1, 0.9, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        centers = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Add jitter
        if jitter_scale > 0:
            centers += rng.uniform(-jitter_scale, jitter_scale, size=centers.shape)
            centers = np.clip(centers, 0.05, 0.95)
        
        # Calculate max radius for each center (box limited)
        max_radii_box = np.min(np.stack([
            centers[:, 0], 
            1.0 - centers[:, 0], 
            centers[:, 1], 
            1.0 - centers[:, 1]
        ], axis=1), axis=1)
        
        # Sort by max radius descending
        sorted_indices = np.argsort(-max_radii_box)
        
        selected_centers = []
        selected_radii = []
        
        for idx in sorted_indices:
            if len(selected_centers) >= N:
                break
            
            c = centers[idx]
            r = max_radii_box[idx]
            
            # Check against already selected
            valid = True
            for k in range(len(selected_centers)):
                dist = np.linalg.norm(c - selected_centers[k])
                # If the new circle cannot fit with positive radius
                if dist - selected_radii[k] <= 0:
                    valid = False
                    break
                r = min(r, dist - selected_radii[k])
            
            if valid and r > 1e-4:
                selected_centers.append(c)
                selected_radii.append(r)
        
        if len(selected_centers) < N:
            return None, 0.0
            
        centers_arr = np.array(selected_centers)
        
        # Refine radii to maximize sum
        refined_radii = refine_radii(centers_arr)
        if refined_radii is None:
            return None, 0.0
            
        score = np.sum(refined_radii)
        return np.column_stack((centers_arr, refined_radii)), score

    # Multi-start search
    seed_counter = 0
    grid_sizes = [10, 12, 15, 20, 25, 30]
    jitter_scales = [0.0, 0.01, 0.02, 0.05]
    
    # Try different combinations
    while not time_up():
        for gs in grid_sizes:
            for js in jitter_scales:
                if time_up():
                    break
                circles, score = greedy_pack(gs, js, seed_counter)
                if circles is not None and score > best_score:
                    best_score = score
                    best_circles = circles
                seed_counter += 1
                # Increment seed for next run
                if seed_counter % 100 == 0 and time_up():
                    break

    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)

    return best_circles

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
