"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def _init_hexagonal(N):
    """
    Initialize points in a hexagonal lattice pattern.
    This provides a much better starting configuration than random rings.
    """
    points = []
    # Approximate number of rings needed
    # Area of hexagon ~ sqrt(3)/2 * d^2
    # We want to fit N points in unit circle.
    # Let's generate a large hex grid and filter.
    
    # Estimate spacing. For N=300, r ~ 0.053. Diameter ~ 0.106.
    # Spacing should be slightly larger than diameter to avoid initial overlap, 
    # or we can start with overlaps and let optimization fix them.
    # Let's use spacing s = 0.11.
    s = 0.11
    h = s * np.sqrt(3) / 2
    
    max_radius = 1.0
    # Generate points in a square grid of hex lattice
    # Range: -max_radius to max_radius
    range_x = int(max_radius / (s * 0.5)) + 5
    range_y = int(max_radius / h) + 5
    
    for i in range(-range_x, range_x + 1):
        for j in range(-range_y, range_y + 1):
            x = i * s + (j % 2) * (s / 2)
            y = j * h
            if x*x + y*y <= max_radius*max_radius:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have too many, randomly subsample or take the ones closest to center?
    # If too few, add random points?
    # Hex lattice is dense. We likely have > 300 points.
    # We want to select N points. 
    # Strategy: Keep the ones that form the best packing? 
    # Simpler: Just take the first N? No, that's biased.
    # Better: Sort by distance from center? No, that leaves holes.
    # Let's just use all points generated if count >= N, and randomly pick N.
    # Or, better, generate a lattice that is slightly sparse so we get close to N.
    
    # Let's adjust s to get closer to N points.
    # Area of unit circle = pi. 
    # Area per point in hex packing = s^2 * sqrt(3)/2.
    # N ~ pi / (s^2 * sqrt(3)/2) => s ~ sqrt(2*pi / (N*sqrt(3)))
    s_est = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    s = s_est * 1.05 # Slightly larger to avoid initial overlaps
    
    h = s * np.sqrt(3) / 2
    range_x = int(max_radius / (s * 0.5)) + 5
    range_y = int(max_radius / h) + 5
    
    points = []
    for i in range(-range_x, range_x + 1):
        for j in range(-range_y, range_y + 1):
            x = i * s + (j % 2) * (s / 2)
            y = j * h
            if x*x + y*y <= max_radius*max_radius:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > N:
        # Randomly select N points
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # If we don't have enough, add random points
        missing = N - len(points)
        new_pts = np.random.uniform(-0.9, 0.9, (missing, 2))
        # Ensure inside
        norms = np.linalg.norm(new_pts, axis=1)
        mask = norms > 0.9
        new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis] * 0.9
        points = np.vstack([points, new_pts])
        
    return points

def _optimize(points, max_iter=5000, step_size=0.01):
    """
    Deterministic Hill Climbing / Gradient Ascent.
    Objective: Maximize r = min(min_pair_dist/2, min_dist_to_boundary)
    
    Strategy:
    1. Compute current score r.
    2. Identify critical points: 
       - Pairs with distance closest to 2*r
       - Points with distance to boundary closest to r
    3. Move critical points in direction that increases their local clearance.
    """
    pts = points.copy()
    
    for it in range(max_iter):
        # Compute current metrics
        norms = np.linalg.norm(pts, axis=1)
        dist_to_bound = 1.0 - norms
        
        # Pairwise distances using KDTree for efficiency
        tree = cKDTree(pts)
        # Find 2 nearest neighbors for each point (one is self)
        dists, indices = tree.query(pts, k=2)
        min_pair_dists = dists[:, 1]
        
        # The limiting radius is determined by:
        # 1. min(min_pair_dists) / 2
        # 2. min(dist_to_bound)
        
        min_pair_min = np.min(min_pair_dists)
        min_bound_min = np.min(dist_to_bound)
        
        current_r = min(min_pair_min / 2.0, min_bound_min)
        
        # If we are stuck or improving is minimal, break? 
        # For now, just run fixed iterations.
        
        # Determine which constraint is active
        # If min_pair_min/2 < min_bound_min, pair constraint is active.
        # Else, boundary constraint is active.
        
        # We want to increase the minimum.
        # Move points that are causing the minimum.
        
        # Identify points involved in the minimum pair distance
        # This is tricky to do exactly for the global min.
        # Approximation: Move points that have small min_pair_dist or small dist_to_bound.
        
        # Threshold for "critical"
        # Let's move all points that are within 10% of the limiting value.
        
        limit_val = current_r
        threshold = limit_val * 0.1 + 1e-6
        
        # Points to move:
        # 1. Those with min_pair_dist < 2*(limit_val + threshold)
        # 2. Those with dist_to_bound < limit_val + threshold
        
        # Actually, let's just compute the "score" for each point locally?
        # No, let's use a simple repulsive force model which is robust.
        
        # Force on point i:
        # 1. Repulsion from neighbors: Sum of (direction / dist^2) for neighbors within 2*r_limit
        # 2. Attraction to center if too close to boundary? Or repulsion from boundary.
        
        # Let's use a gradient step based on the active constraints.
        # For each point i:
        # grad_r_i is complex because of min.
        # Instead, let's use the "Pushing" method.
        
        # For each pair (i, j) with dist < 2 * (current_r + epsilon):
        # Push i and j apart.
        # For each point i with dist_to_bound < current_r + epsilon:
        # Push i inward.
        
        eps = 0.001
        limit_dist = 2.0 * (current_r + eps)
        limit_bound = current_r + eps
        
        # Build forces
        forces = np.zeros_like(pts)
        
        # Pair repulsion
        # We only need to check close pairs.
        # Use KDTree to find pairs within limit_dist
        # Note: tree.query_pairs gives unique pairs
        
        # To be efficient, we can just use the 2 nearest neighbors we already found?
        # No, there might be more than 2 close neighbors in a dense packing.
        # Let's query all neighbors within limit_dist.
        pairs = tree.query_pairs(r=limit_dist)
        
        for i, j in pairs:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-9:
                # Repulsive force
                # Strength decays with distance? Or constant?
                # We want to increase the minimum.
                # If dist < 2*current_r, we MUST increase it.
                # Force direction: d / dist
                # Magnitude: proportional to (2*current_r - dist) if dist < 2*current_r, else small.
                # Let's use a linear potential for violations.
                if dist < 2.0 * current_r:
                    mag = (2.0 * current_r - dist) / (2.0 * current_r)
                else:
                    mag = 0.01 # Small repulsion to maintain spacing
                forces[i] += d / dist * mag
                forces[j] -= d / dist * mag
                
        # Boundary repulsion
        for i in range(len(pts)):
            dist_b = dist_to_bound[i]
            if dist_b < limit_bound:
                # Push inward
                # Direction: -pts[i] / norm(pts[i])
                norm_i = norms[i]
                if norm_i > 1e-9:
                    dir_in = -pts[i] / norm_i
                    # Magnitude: proportional to violation
                    if dist_b < current_r:
                        mag = (current_r - dist_b) / current_r
                    else:
                        mag = 0.01
                    forces[i] += dir_in * mag
        
        # Apply forces
        # Step size should be small to avoid oscillation
        # Normalize forces?
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-9:
            forces = forces / max_force * step_size
            
        pts += forces
        
        # Clamp to unit circle
        norms_new = np.linalg.norm(pts, axis=1)
        mask = norms_new > 0.999999
        if np.any(mask):
            pts[mask] = pts[mask] / norms_new[mask, np.newaxis] * 0.999999

    return pts

def _score(points):
    """
    Calculate the score: min(min_pair_dist/2, min_dist_to_boundary)
    """
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(norms)
    
    tree = cKDTree(points)
    # Find 2 nearest neighbors
    dists, indices = tree.query(points, k=2)
    min_pair_dist = np.min(dists[:, 1])
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    best_points = None
    best_score = -1.0
    
    # Run multiple trials with different initializations or step sizes
    # Trial 1: Hex init
    init_pts = _init_hexagonal(N)
    # Add small noise to break symmetry
    init_pts += np.random.normal(0, 0.001, size=init_pts.shape)
    
    # Ensure inside
    norms = np.linalg.norm(init_pts, axis=1)
    mask = norms > 0.999999
    if np.any(mask):
        init_pts[mask] = init_pts[mask] / norms[mask, np.newaxis] * 0.999999
        
    optimized = _optimize(init_pts, max_iter=2000, step_size=0.005)
    score = _score(optimized)
    
    if score > best_score:
        best_score = score
        best_points = optimized.copy()
        
    # Trial 2: Different step size / more iterations
    init_pts2 = _init_hexagonal(N)
    init_pts2 += np.random.normal(0, 0.001, size=init_pts2.shape)
    norms = np.linalg.norm(init_pts2, axis=1)
    mask = norms > 0.999999
    if np.any(mask):
        init_pts2[mask] = init_pts2[mask] / norms[mask, np.newaxis] * 0.999999
        
    optimized2 = _optimize(init_pts2, max_iter=3000, step_size=0.002)
    score2 = _score(optimized2)
    
    if score2 > best_score:
        best_score = score2
        best_points = optimized2.copy()
        
    # Final polish on the best
    # Use a very small step size
    final_optimized = _optimize(best_points, max_iter=1000, step_size=0.0005)
    final_score = _score(final_optimized)
    
    if final_score > best_score:
        best_points = final_optimized
        best_score = final_score
        
    # Ensure validity
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999999
        
    return best_points
# EVOLVE-BLOCK-END
