import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Helper to compute the exact radius
    def compute_radius(c):
        # Pairwise distances
        # Using broadcasting for vectorization, but for N=100 this is 100x100x3
        # which is very fast.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        # Distance from each center to the nearest wall
        # Cube is [-0.5, 0.5]^3
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    # Helper to find the "bottleneck" indices
    # Returns list of indices that are part of the minimum distance constraint
    def get_bottlenecks(c, r):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Find pairs with distance close to 2*r
        # We use a small epsilon to capture the tightest constraints
        eps = 1e-5
        tight_pairs = np.argwhere(dists <= 2.0 * r + eps)
        
        # Also check walls
        wall_dists = 0.5 - np.abs(c)
        tight_walls = np.argwhere(wall_dists <= r + eps)
        
        bottleneck_indices = set()
        for i, j in tight_pairs:
            bottleneck_indices.add(i)
            bottleneck_indices.add(j)
        
        for i, k in tight_walls:
            bottleneck_indices.add(i)
            
        return list(bottleneck_indices)

    # Strategy:
    # 1. Generate a candidate lattice spacing 's'.
    # 2. Place points on a cubic lattice with spacing 's' centered at 0.
    # 3. Select the N points that form the densest cluster around the center 
    #    (or just take the first N in a sorted order of distance from center).
    # 4. Clip to cube.
    # 5. Perform local refinement on the bottleneck spheres using Nelder-Mead.
    
    best_centers = None
    best_radius = 0.0
    
    # Search space for lattice spacing 's'
    # If we have a ~4x4x4 or 5x5x4 grid, s is roughly 1/sqrt(N) ~ 0.1
    # But we want to maximize radius, so we want to pack as many as possible.
    # The optimal s is determined by the boundary.
    # Let's scan a range of s values.
    
    # Estimate optimal s:
    # Volume of sphere ~ s^3. Total volume ~ N * s^3 <= 1.
    # s <= N^(-1/3) ~ 0.215
    # However, packing density is < 1.
    # Let's scan s from 0.15 to 0.25
    
    s_values = np.linspace(0.15, 0.25, 20)
    
    for s in s_values:
        if time.time() > deadline - 30:
            break
            
        # Generate lattice points
        # Number of points in one dimension
        n_dim = int(np.ceil(1.0 / s)) + 1
        # Create grid
        coords = np.arange(-n_dim//2, n_dim//2 + 1) * s
        grid = np.array([(x, y, z) for z in coords for y in coords for x in coords])
        
        # Sort by distance to center to pick the central cluster
        dists_to_center = np.linalg.norm(grid, axis=1)
        sorted_indices = np.argsort(dists_to_center)
        
        # We need N points. 
        # If the grid has fewer than N points, this s is too large or grid is too small.
        if len(grid) < N:
            continue
            
        # Take the first N central points
        centers = grid[sorted_indices[:N]].copy()
        
        # Ensure they are within the cube [-0.5, 0.5]
        # If a point is outside, we might need to adjust s or accept clipping (which reduces radius)
        # Better: only consider points that are strictly inside with some margin?
        # Actually, the radius calculation handles walls.
        # But if a center is outside, it's invalid.
        # So we must ensure centers are within [-0.5, 0.5].
        # If we clip, the effective radius drops.
        # Let's check if the selected points are all inside.
        if np.any(np.abs(centers) > 0.5):
            # Try to find N points that are inside
            inside_mask = np.all(np.abs(grid) <= 0.5, axis=1)
            if np.sum(inside_mask) < N:
                continue
            # Re-select from inside points only
            inside_grid = grid[inside_mask]
            dists_inside = np.linalg.norm(inside_grid, axis=1)
            sorted_inside = np.argsort(dists_inside)
            centers = inside_grid[sorted_inside[:N]].copy()
        
        # Initial radius
        r = compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        # Local Refinement Phase
        # We want to improve the radius by moving the spheres that are "tight".
        # We will perform a few iterations of:
        # 1. Identify bottleneck spheres.
        # 2. For each bottleneck sphere, perform a local Nelder-Mead optimization
        #    maximizing the global radius, but restricting the movement to be small.
        #    Actually, Nelder-Mead on the full N=100 system is too slow (100 dimensions).
        #    So we optimize one sphere at a time (3 dimensions).
        
        # To speed up, we limit the number of refinement steps.
        max_refine_steps = 20
        step_count = 0
        
        while step_count < max_refine_steps and time.time() < deadline - 10:
            r = compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                
            bottlenecks = get_bottlenecks(centers, r)
            if not bottlenecks:
                break
                
            # Shuffle bottlenecks to avoid bias
            np.random.shuffle(bottlenecks)
            
            improved = False
            for idx in bottlenecks:
                if time.time() > deadline - 10:
                    break
                    
                original_pos = centers[idx].copy()
                
                # Objective: Maximize radius. 
                # We define a function to minimize -radius.
                # However, moving one sphere affects the radius of the whole system.
                # We can use a simple line search or Nelder-Mead in 3D.
                
                def objective(pos):
                    # Temporarily update center
                    new_centers = centers.copy()
                    new_centers[idx] = pos
                    # Check bounds
                    if np.any(np.abs(new_centers) > 0.5):
                        return 1e9 # Invalid
                    return -compute_radius(new_centers)
                
                # Nelder-Mead
                # Initial simplex: small perturbations
                x0 = original_pos
                simplex = [x0]
                for d in range(3):
                    pert = x0.copy()
                    pert[d] += 0.01
                    simplex.append(pert)
                
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 100, 'fatol': 1e-6, 'xatol': 1e-6})
                
                if res.fun < objective(original_pos) - 1e-8:
                    centers[idx] = res.x
                    improved = True
                    
            if not improved:
                break
            step_count += 1

    # Final check
    if best_centers is None:
        # Fallback: Random placement
        rng = np.random.default_rng(42)
        best_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

if __name__ == "__main__":
    c = construct()
    print("Radius:", compute_radius(c))
