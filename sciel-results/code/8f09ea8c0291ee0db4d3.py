"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # 1. Initialization: Jittered Grid
    # 50 spheres. 4x4x4 grid is 64 points. We take 50.
    # Better: 5x5x2 is 50 points exactly.
    # Let's try a 5x5x2 grid centered in the cube.
    # Dimensions: x in [-0.5, 0.5], y in [-0.5, 0.5], z in [-0.5, 0.5]
    # 5 divisions in x, 5 in y, 2 in z.
    nx, ny, nz = 5, 5, 2
    points = []
    # We want to maximize the minimum distance.
    # A regular grid is a good start.
    # Spacing in x: 1/5 = 0.2
    # Spacing in y: 1/5 = 0.2
    # Spacing in z: 1/2 = 0.5
    # This leaves large gaps in z.
    # Alternative: 4x4x4 = 64. Remove 14.
    # Let's stick to 5x5x2 but shift the z-layers to be closer? No, uniform is better for init.
    # Actually, let's use a 4x4x4 grid and randomly select 50? No, deterministic is better for reproducibility.
    # 5x5x2 grid:
    xs = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    ys = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    zs = np.linspace(-0.5 + 0.5/2, 0.5 - 0.5/2, 2)
    
    grid_points = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid_points.append([x, y, z])
    centers = np.array(grid_points)
    
    # Add small random jitter to break symmetry and help escape local minima
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.001, centers.shape)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Lloyd's Algorithm (Voronoi Relaxation)
    # Iterate: Compute Voronoi cells, find centroid of each cell (clipped to cube), update centers.
    # We need to handle the fact that Voronoi cells extend to infinity, so we clip to the cube [-0.5, 0.5]^3.
    
    max_iters = 500
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        try:
            vor = Voronoi(centers)
        except Exception:
            # Voronoi can fail if points are collinear or coincident, though unlikely with jitter
            break
            
        new_centers = np.zeros_like(centers)
        
        for idx in range(N):
            # Get the region for point idx
            region_idx = vor.point_region[idx]
            region = vor.regions[region_idx]
            
            if -1 in region:
                # Unbounded region? In a bounded box, we should clip.
                # However, scipy's Voronoi gives infinite regions.
                # We must intersect the infinite polytope with the cube.
                # This is complex. 
                # Alternative: Use a "soft" Voronoi or just move towards the average of neighbors?
                # A simpler robust approximation for Lloyd in a box:
                # Move center towards the centroid of its "influence zone".
                # If we can't easily clip infinite Voronoi, we can use a different relaxation:
                # Move away from nearest neighbors.
                
                # Let's try a simpler heuristic if Voronoi clipping is too hard/slow:
                # Repulsion-based relaxation.
                # But Lloyd is better.
                # To clip Voronoi to a box, we can define the box faces as additional "points" at infinity? No.
                # We can compute the intersection of the Voronoi polytope (defined by vertices) with the cube.
                # scipy.spatial.Voronoi provides vertices.
                # The region is defined by vertices.
                # If the region is unbounded, it has no vertices in scipy's representation?
                # Actually, scipy returns empty region for unbounded, or vertices for bounded.
                # For points inside a box, most regions will be bounded if the points are dense enough.
                # But boundary points will have unbounded regions.
                
                # Workaround: If region is unbounded, we can't easily get the centroid.
                # Let's use a different method: **Repulsive Force Relaxation**.
                # It's simpler and robust.
                # Move each point away from its k-nearest neighbors.
                pass
        
        # Since Voronoi clipping is tricky with scipy's default output (unbounded regions),
        # let's switch to a **Repulsive Force Relaxation** which is effectively a gradient step
        # on the sum of inverse distances, projected to the box.
        # This is often sufficient to find good packings.
        
        # Calculate forces
        forces = np.zeros_like(centers)
        for i in range(N):
            # Vector to all other points
            diffs = centers - centers[i]
            dists = np.linalg.norm(diffs, axis=1)
            # Avoid division by zero
            dists[i] = 1.0
            # Force magnitude: 1/d^2 (repulsion)
            # We want to maximize min dist, so we push apart.
            # Gradient of min dist is complex, but sum of 1/d^2 works well for relaxation.
            weights = 1.0 / (dists ** 2)
            weights[i] = 0.0
            forces[i] = np.sum(diffs * weights[:, np.newaxis], axis=0)
        
        # Update centers
        # Step size needs to be small
        step_size = 0.01
        new_centers = centers + step_size * forces
        
        # Project to cube
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Check for overlap (if any, reduce step size or revert)
        # Simple check: if min dist decreased significantly, revert
        new_radius = _compute_radius(new_centers)
        if new_radius < best_radius * 0.9:
             # Bad move, try smaller step or revert
             new_centers = centers + 0.001 * forces
             new_centers = np.clip(new_centers, -0.5, 0.5)
             new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers
        else:
            # Even if not better, accept if close? No, keep best.
            # But for relaxation, we usually follow the force.
            # Let's follow the force but keep track of best.
            pass
            
        centers = new_centers
        
        # Convergence check
        if i > 10 and abs(new_radius - best_radius) < 1e-8:
             # Stuck, try a random perturbation
             centers += rng.normal(0, 0.01, centers.shape)
             centers = np.clip(centers, -0.5, 0.5)
             
    # Final Polish: Local Refinement
    # Use a simple hill-climbing step to maximize the minimum distance
    # We will try to move each point slightly to increase the min distance to its neighbors.
    
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Find the neighbor that is closest
            dists = np.linalg.norm(best_centers - best_centers[i], axis=1)
            dists[i] = np.inf
            j = np.argmin(dists)
            min_d = dists[j]
            
            # Direction to move i away from j
            dir_vec = best_centers[i] - best_centers[j]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-9:
                dir_vec /= norm
            
            # Try moving i in direction dir_vec
            # Also check walls
            # Wall distance for i
            wall_dists = np.minimum(0.5 - np.abs(best_centers[i]), 0.5 - np.abs(best_centers[i])) # Wait, min(0.5-x, 0.5+x) is 0.5-|x|
            wall_dists = 0.5 - np.abs(best_centers[i])
            min_wall = np.min(wall_dists)
            
            # If the bottleneck is the wall, move towards center
            # If the bottleneck is the neighbor, move away
            # Let's just try a small step in dir_vec
            step = 0.001
            new_c = best_centers[i] + step * dir_vec
            new_c = np.clip(new_c, -0.5, 0.5)
            
            # Check if this improves the global min radius
            # Only check against j and walls for speed
            new_d_j = np.linalg.norm(new_c - best_centers[j])
            new_wall = np.min(0.5 - np.abs(new_c))
            
            # The global min radius is min(all pairs, all walls)
            # We are approximating. If new_d_j > min_d and new_wall > min_wall, it's likely good.
            # But it might hurt other pairs.
            # For safety, compute full radius for the new candidate? Too slow for 50*50*50.
            # Let's just compute the radius for the whole system if we accept.
            
            candidate = best_centers.copy()
            candidate[i] = new_c
            r_new = _compute_radius(candidate)
            
            if r_new > best_radius:
                best_radius = r_new
                best_centers = candidate
                improved = True
        if not improved:
            break
            
    return best_centers

def _compute_radius(centers):
    """Compute the maximum radius for the given centers."""
    N = len(centers)
    min_dist = np.inf
    
    # Pairwise distances
    # Use broadcasting for speed
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # This creates an NxNx3 array, which is 50*50*3 = 7500 floats. Very fast.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    
    # Min pair distance
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # Distance to walls: min(0.5 - x, 0.5 + x) = 0.5 - |x|
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half the min distance
    r = min(min_pair_dist, min_wall_dist) / 2.0
    
    return r

# EVOLVE-BLOCK-END
