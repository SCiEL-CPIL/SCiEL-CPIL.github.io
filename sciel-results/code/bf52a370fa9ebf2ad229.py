import numpy as np
import time

N = 100

def construct():
    """
    Constructs the centers of 100 equal spheres in a unit cube [-0.5, 0.5]^3.
    Uses a greedy initialization from a dense grid followed by local optimization.
    """
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # 1. Generate a dense grid of candidate points
    # Spacing of 0.05 gives 20x20x20 = 8000 points. 
    # This is fine enough to capture the optimal spacing (~0.22).
    grid_spacing = 0.05
    m = int(np.floor(1.0 / grid_spacing)) + 1
    ticks = np.linspace(-0.5 + grid_spacing/2, 0.5 - grid_spacing/2, m)
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # 2. Greedy Selection
    # Select N points from the grid to maximize the minimum distance.
    selected = []
    remaining = list(range(len(grid)))
    
    # Heuristic: Start with a point far from boundaries? Or just pick one.
    # Let's pick the point that maximizes distance to boundary first? 
    # Or just pick the center.
    # Actually, a good start is to pick points that are well-separated.
    
    # Initialize with one point (e.g., center or first point)
    # Let's pick the point closest to the center to start, then add others.
    # Or better: Randomly pick a few, then greedy.
    # To be deterministic, we use a fixed order or a deterministic heuristic.
    
    # Let's use a "max-min" greedy selection.
    # Step 1: Pick the point with max distance to boundary? No, that's just center.
    # Let's pick the first point as the center of the cube (closest to 0,0,0 in grid).
    center_idx = np.argmin(np.sum(grid**2, axis=1))
    selected.append(grid[center_idx])
    remaining.remove(center_idx)
    
    for _ in range(N - 1):
        if not remaining:
            break
        if time.time() - start_time > timeout:
            break
            
        best_idx = -1
        best_min_dist = -1
        
        # For each remaining candidate, calculate the min distance to already selected points
        # and to the boundary.
        # We want to maximize this min distance.
        
        # Vectorized calculation for speed
        # Candidates: grid[remaining]
        # Selected: np.array(selected)
        
        candidates = grid[remaining]
        sel = np.array(selected)
        
        # Distances between candidates and selected points
        # dists[i, j] = |candidates[i] - sel[j]|
        # This is (len(remaining) x len(selected))
        # We can do this in chunks if memory is an issue, but 8000x100 is small.
        
        # Calculate squared distances
        # candidates: (M, 3)
        # sel: (K, 3)
        # diff: (M, 1, 3) - (1, K, 3) -> (M, K, 3)
        # This might be large: 8000 * 100 * 3 = 2.4M floats. OK.
        
        diff = candidates[:, np.newaxis, :] - sel[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dists_to_sel = np.min(dists, axis=1)
        
        # Distances to boundary
        # Boundary dist for a point p is min(0.5 - |p_x|, 0.5 - |p_y|, 0.5 - |p_z|)
        # Actually, the constraint is 0.5 - |p_k| >= r.
        # So dist to boundary is min_k (0.5 - |p_k|)
        abs_cands = np.abs(candidates)
        dists_to_bound = np.min(0.5 - abs_cands, axis=1)
        
        # The effective distance for a candidate is the min of its distance to other spheres 
        # and its distance to the boundary.
        # But wait, if we place a sphere here, its radius is limited by 
        # min(min_dists_to_sel / 2, dists_to_bound).
        # We want to maximize the *global* minimum radius.
        # A greedy strategy is to pick the candidate that maximizes the *local* min distance.
        
        local_min_dists = np.minimum(min_dists_to_sel, dists_to_bound)
        
        # Pick the candidate with the maximum local_min_dist
        max_idx_in_remaining = np.argmax(local_min_dists)
        best_idx = remaining[max_idx_in_remaining]
        best_min_dist = local_min_dists[max_idx_in_remaining]
        
        selected.append(grid[best_idx])
        remaining.remove(best_idx)
        
    centers = np.array(selected)
    
    # 3. Local Optimization
    # Refine the positions to increase the minimum radius.
    # Use a simple iterative improvement.
    
    def compute_radius(centers):
        # Min distance between pairs / 2
        # Min distance to boundary
        if len(centers) < 2:
            return 1.0
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        abs_c = np.abs(centers)
        dists_to_bound = np.min(0.5 - abs_c, axis=1)
        min_bound_dist = np.min(dists_to_bound)
        
        return min(min_pair_dist / 2.0, min_bound_dist)
    
    # Simple Hill Climbing
    # Move points slightly in random directions or towards "free space".
    # To be deterministic, we can use a fixed sequence of small perturbations.
    
    current_radius = compute_radius(centers)
    
    # We will try to increase the radius by moving points.
    # Strategy: For each point, find the direction that increases the minimum distance to its neighbors/boundary.
    
    for step in range(500): # Limit iterations to ensure time safety
        if time.time() - start_time > timeout:
            break
            
        # Compute gradients? Or just perturb.
        # Perturbation method:
        # For each point i, try moving it by a small vector v.
        # Accept if radius increases.
        
        # To make it faster, we only consider the "critical" points.
        # But for simplicity, let's do a few passes of random perturbations.
        
        # Deterministic pseudo-random sequence
        # Use a simple LCG for reproducibility
        state = 12345
        def rand():
            nonlocal state
            state = (state * 1103515245 + 12345) % (2**31)
            return state / (2**31)
        
        # Step size decay
        step_size = 0.01
        improved = False
        
        for i in range(N):
            if time.time() - start_time > timeout:
                break
            
            # Try moving point i in a random direction
            # Generate a random vector
            v = np.array([rand() - 0.5, rand() - 0.5, rand() - 0.5])
            v = v / np.linalg.norm(v) * step_size
            
            new_centers = centers.copy()
            new_centers[i] = centers[i] + v
            
            # Check if still inside cube
            if np.any(np.abs(new_centers[i]) > 0.5):
                continue
                
            new_radius = compute_radius(new_centers)
            
            if new_radius > current_radius + 1e-12:
                centers = new_centers
                current_radius = new_radius
                improved = True
            else:
                # Try opposite direction?
                new_centers[i] = centers[i] - v
                if np.any(np.abs(new_centers[i]) > 0.5):
                    continue
                new_radius = compute_radius(new_centers)
                if new_radius > current_radius + 1e-12:
                    centers = new_centers
                    current_radius = new_radius
                    improved = True
                    
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-5:
                break
                
    return centers

# EVOLVE-BLOCK-END
