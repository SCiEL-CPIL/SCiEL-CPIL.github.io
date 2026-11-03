"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def _hexagonal_lattice(N_target):
    """
    Generate a hexagonal lattice of points inside the unit circle.
    We generate a grid and then select the N_target points closest to the center
    or just take the first N_target points from a sorted list by distance from center?
    Actually, for packing, we want uniform density.
    Let's generate a large hex grid and keep points within radius 1.
    Then if we have more than N_target, we remove the ones with the smallest "room" (distance to neighbors/boundary) or just randomly remove?
    Better: Generate a hex grid with spacing roughly 1/sqrt(N_target).
    """
    # Approximate spacing
    # Area per circle ~ pi * r^2. Total area pi. N circles.
    # r ~ 1/sqrt(N). Spacing ~ 2r ~ 2/sqrt(N).
    spacing = 2.0 / np.sqrt(N_target)
    
    points = []
    # Range of x, y
    limit = 1.2
    x_vals = np.arange(-limit, limit, spacing)
    y_vals = np.arange(-limit, limit, spacing * np.sqrt(3)/2)
    
    for i, y in enumerate(y_vals):
        for x in x_vals:
            # Shift every other row by half spacing for hex packing
            if i % 2 == 1:
                x += spacing / 2.0
            
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < N_target:
        # If we don't have enough, add random points inside
        while len(points) < N_target:
            angle = np.random.uniform(0, 2*np.pi)
            r = np.sqrt(np.random.uniform(0, 1))
            points = np.vstack([points, [r*np.cos(angle), r*np.sin(angle)]])
    else:
        # If we have too many, we need to select N_target.
        # To maintain good packing, we should probably keep the ones that are most "central" or have largest local space?
        # For simplicity, let's just take the first N_target. 
        # But a better heuristic: sort by distance to origin and take the inner ones? No, that clumps them.
        # Actually, a hex grid clipped to a circle is already a very good packing.
        # If we have too many, we can remove points that are closest to the boundary?
        # Or just remove random ones?
        # Let's remove the points that are furthest from the origin? No, that leaves a hole in the middle.
        # Let's remove points randomly to break symmetry, but ensure we have N_target.
        indices = np.random.choice(len(points), N_target, replace=False)
        points = points[indices]
        
    return points

def _local_search(points, max_iters=2000, initial_step=0.01, cooling=0.999, noise_scale=0.001):
    """
    Perform local search using repulsive forces.
    Objective: Maximize min distance.
    Proxy: Minimize sum of 1/d^2 (repulsion).
    """
    pts = points.copy()
    step_size = initial_step
    
    for _ in range(max_iters):
        # Compute pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Repulsive force: F_i = sum_j (p_i - p_j) / d_ij^2
        # Vectorized:
        # inv_dist_sq = 1 / dist_sq
        # Force[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Boundary force: Push points back if they are outside or too close to edge?
        # The problem requires points to be INSIDE.
        # We can add a force that pushes points towards the center if they are near the boundary.
        # Or simply clamp them after the step.
        # Clamping is safer for validity.
        
        # Add noise to escape local minima
        noise = np.random.normal(0, noise_scale, size=pts.shape)
        
        # Update
        pts = pts + step_size * forces + noise
        
        # Clamp to unit circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999999
            
        # Cool down
        step_size *= cooling
        noise_scale *= 0.999
        
        if step_size < 1e-8:
            break
            
    return pts

def _score(points):
    """
    Calculate the score: min(min_pair_dist/2, min_dist_to_boundary)
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(norms)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    best_points = None
    best_score = -1.0
    
    # Strategy 1: Start with Hexagonal Lattice
    # Generate a few variations of hex lattice with slight random shifts
    for i in range(5):
        pts = _hexagonal_lattice(N)
        # Add small random perturbation to break symmetry
        pts += np.random.normal(0, 0.001, size=pts.shape)
        
        # Run local search
        optimized = _local_search(pts, max_iters=1000, initial_step=0.005, cooling=0.9995, noise_scale=0.0005)
        
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
    # Strategy 2: Start with Random Points (might find different local minima)
    for i in range(5):
        # Random points in unit disk
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Run local search with higher noise to explore
        optimized = _local_search(pts, max_iters=2000, initial_step=0.01, cooling=0.999, noise_scale=0.002)
        
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
    # Final polish: Very fine local search on the best result
    if best_points is not None:
        final_optimized = _local_search(best_points, max_iters=500, initial_step=0.001, cooling=0.9999, noise_scale=0.0)
        final_score = _score(final_optimized)
        if final_score > best_score:
            best_points = final_optimized
            best_score = final_score

    # Ensure validity: strictly inside
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999999
        
    return best_points
# EVOLVE-BLOCK-END
