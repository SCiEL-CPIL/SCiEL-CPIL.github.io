"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _lloyd_iteration(centers, max_iter=20, time_limit=10.0):
    """
    Lloyd's algorithm variant: move each center to the centroid of its Voronoi cell
    clipped by the cube boundary. This equalizes the influence regions, pushing
    centers away from walls and neighbors to maximize the minimum clearance.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Add ghost points outside the cube to clip Voronoi cells
    # We add points on the faces, edges, and corners slightly outside
    ghosts = []
    # Faces
    for axis in range(3):
        for val in [-0.51, 0.51]:
            for i in range(5):
                for j in range(5):
                    g = np.array([0.0, 0.0, 0.0])
                    g[axis] = val
                    # Add some spread to better approximate the wall constraint
                    # Actually, for Voronoi clipping, we just need points outside.
                    # A simpler approach: use the cube corners and face centers as ghosts?
                    # No, standard Voronoi with points outside the domain works.
                    # Let's just add a grid of points just outside the cube.
                    pass
    
    # Better ghost strategy: Add points at the corners and face centers of a slightly larger cube
    # Or simply: The Voronoi cell of a point p is the set of points closer to p than any other.
    # If we don't add ghosts, the cells of boundary points extend to infinity.
    # We want to clip them at the cube boundary.
    # We can simulate this by adding a dense set of points just outside the cube.
    
    ghost_points = []
    # Add points on the 6 faces, just outside
    for axis in range(3):
        for side in [-1, 1]:
            val = side * 0.51
            for u in np.linspace(-0.5, 0.5, 4):
                for v in np.linspace(-0.5, 0.5, 4):
                    g = np.zeros(3)
                    g[axis] = val
                    # Fill other two coords with u, v
                    other_axes = [a for a in range(3) if a != axis]
                    g[other_axes[0]] = u
                    g[other_axes[1]] = v
                    ghost_points.append(g)
    
    ghosts = np.array(ghost_points)
    all_points = np.vstack([centers, ghosts])
    
    # Compute Voronoi
    try:
        vor = Voronoi(all_points)
    except Exception:
        return centers
    
    # For each original point (index < N), compute centroid of its Voronoi cell
    # Clipped to the cube [-0.5, 0.5]^3
    
    new_centers = centers.copy()
    
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region, skip or handle carefully
            continue
            
        vertices = vor.vertices[region]
        
        # Clip vertices to the cube
        # We need to compute the centroid of the polygon (intersection of Voronoi cell and Cube)
        # This is complex in 3D. 
        # Approximation: If the Voronoi cell is large, the centroid might be outside.
        # A simpler heuristic for Lloyd's in a bounded domain:
        # If the centroid of the unclipped cell is outside the cube, project it back?
        # Or, use the fact that for boundary points, the "force" should push them inward.
        
        # Let's try a simpler heuristic first:
        # Calculate centroid of the Voronoi vertices.
        if len(vertices) > 0:
            centroid = np.mean(vertices, axis=0)
            
            # If centroid is outside, it suggests the point is on the boundary.
            # We want to push it towards the center.
            # But we also need to respect the repulsion from neighbors.
            
            # Let's just use the centroid, but clip it to the cube.
            # This might not be optimal but is a valid step.
            new_center = np.clip(centroid, -0.49, 0.49)
            
            # Blend with original position to avoid instability
            new_centers[i] = 0.5 * centers[i] + 0.5 * new_center
    
    # After Lloyd step, we must ensure no overlaps and points are inside.
    # Lloyd's tends to equalize distances, but might cause overlaps if started poorly.
    # We should run a local search to fix overlaps and maximize min distance.
    
    return new_centers

def _local_search(centers, time_limit=5.0):
    """
    Direct local search maximizing the minimum distance.
    Perturbs spheres involved in the current minimum distance constraints.
    """
    start_time = time.time()
    centers = centers.copy()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, k=1)
    step_size = 0.02
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute min pair and min wall
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        pair_dists_sq = dists_sq[iu]
        min_pair_dist_sq = np.min(pair_dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist_per_sphere = np.min(wall_dists, axis=1)
        global_min_wall_dist = np.min(min_wall_dist_per_sphere)
        
        current_r = min(min_pair_dist / 2.0, global_min_wall_dist)
        
        # Find critical indices
        min_pair_idx = np.argmin(pair_dists_sq)
        i, j = iu[0][min_pair_idx], iu[1][min_pair_idx]
        min_wall_idx = np.argmin(min_wall_dist_per_sphere)
        
        candidates = [i, j, min_wall_idx]
        # Add random sphere
        if np.random.rand() < 0.1:
            candidates.append(np.random.randint(N))
        
        improved = False
        for idx in candidates:
            # Try to move idx
            for _ in range(10): # 10 random trials per candidate
                direction = np.random.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                new_pos = centers[idx] + direction * step_size
                
                if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                    continue
                    
                old_pos = centers[idx]
                centers[idx] = new_pos
                
                new_score = _score(centers)
                
                if new_score > current_r:
                    current_r = new_score
                    if current_r > best_score:
                        best_score = current_r
                        best_centers = centers.copy()
                    improved = True
                    break # Accept this move, try next candidate
                else:
                    centers[idx] = old_pos
            
            if improved:
                step_size *= 0.9 # Refine
            else:
                step_size *= 1.05 # Explore
                step_size = min(step_size, 0.1)
                
        # Reset step size occasionally
        if iteration % 50 == 0:
             step_size = 0.02

    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Strategy: Mix of Grid Pruning and Random, followed by Lloyd's and Local Search
    
    # 1. Initializations
    # Grid Pruning
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Random
    # centers = rng.uniform(-0.45, 0.45, (N, 3))
    
    for restart in range(15):
        if time.time() > deadline:
            break
            
        if restart < 8:
            # Grid Pruning
            idx = rng.permutation(len(grid))[:N]
            centers = grid[idx].copy()
            centers += rng.normal(0, 0.01, centers.shape)
        else:
            # Random
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            
        centers = np.clip(centers, -0.49, 0.49)
        
        # 2. Lloyd's Iteration (Voronoi-based)
        # This step equalizes the space, pushing points away from walls and each other.
        # It is slower than local search but provides a better structural arrangement.
        time_left = deadline - time.time()
        if time_left > 20:
            lloyd_time = min(5.0, time_left / 3.0)
            centers = _lloyd_iteration(centers, max_iter=10, time_limit=lloyd_time)
            centers = np.clip(centers, -0.49, 0.49)
        
        # 3. Local Search
        time_left = deadline - time.time()
        ls_time = min(5.0, time_left / 2.0)
        
        refined_centers, score = _local_search(centers, time_limit=ls_time)
        
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()
            
    # Final Polish on the best candidate
    if best_centers is not None and (deadline - time.time()) > 10:
        final_time = deadline - time.time() - 5
        refined_centers, score = _local_search(best_centers, time_limit=final_time)
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
