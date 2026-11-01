"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _fibonacci_sphere(n, radius=1.0):
    """Generate n points uniformly distributed on a sphere using Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(n):
        z = 1 - 2 * i / (n - 1) if n > 1 else 0
        r = np.sqrt(max(0, 1 - z**2))
        theta = 2 * np.pi * i / golden_ratio
        points[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
    
    return points

def _local_search_sphere(centers, time_limit=5.0):
    """
    Local search that keeps points on or near a sphere, optimizing both
    inter-sphere distances and wall distances.
    """
    start_time = time.time()
    centers = centers.copy()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, k=1)
    step_size = 0.015
    sphere_radius = np.linalg.norm(centers, axis=1).mean()
    
    for iteration in range(3000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute constraints
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        pair_dists_sq = dists_sq[iu]
        min_pair_dist_sq = np.min(pair_dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist_per_sphere = np.min(wall_dists, axis=1)
        global_min_wall_dist = np.min(min_wall_dist_per_sphere)
        
        current_r = min(min_pair_dist / 2.0, global_min_wall_dist)
        
        # Find critical constraints
        min_pair_idx = np.argmin(pair_dists_sq)
        i, j = iu[0][min_pair_idx], iu[1][min_pair_idx]
        min_wall_idx = np.argmin(min_wall_dist_per_sphere)
        
        # Target the most constrained spheres
        candidates = list(set([i, j, min_wall_idx]))
        if np.random.rand() < 0.15:
            candidates.append(np.random.randint(N))
        
        improved = False
        for idx in candidates:
            for trial in range(8):
                # Try radial and tangential perturbations
                if np.random.rand() < 0.3:
                    # Radial move (toward/away from origin)
                    direction = -centers[idx] / (np.linalg.norm(centers[idx]) + 1e-10)
                    if np.random.rand() < 0.5:
                        direction = -direction
                else:
                    # Tangential move
                    random_dir = np.random.normal(0, 1, 3)
                    # Make perpendicular to radius
                    radial = centers[idx] / (np.linalg.norm(centers[idx]) + 1e-10)
                    random_dir -= np.dot(random_dir, radial) * radial
                    if np.linalg.norm(random_dir) < 1e-10:
                        continue
                    direction = random_dir / np.linalg.norm(random_dir)
                
                new_pos = centers[idx] + direction * step_size
                
                # Clip to cube
                new_pos = np.clip(new_pos, -0.495, 0.495)
                
                old_pos = centers[idx]
                centers[idx] = new_pos
                
                new_score = _score(centers)
                
                if new_score > current_r:
                    current_r = new_score
                    if current_r > best_score:
                        best_score = current_r
                        best_centers = centers.copy()
                    improved = True
                    break
                else:
                    centers[idx] = old_pos
            
            if improved:
                step_size *= 0.92
            else:
                step_size *= 1.08
                step_size = min(step_size, 0.05)
        
        if iteration % 40 == 0:
            step_size = 0.015

    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Fibonacci sphere initialization
    # Place points on a sphere inscribed in the cube
    # Sphere radius: cube half-side is 0.5, so max sphere radius is 0.5
    # But we need room for sphere radius r, so center sphere radius = 0.5 - r
    # Start with sphere radius ~0.35
    
    sphere_radius = 0.35
    base_points = _fibonacci_sphere(N, radius=sphere_radius)
    
    # Multiple restarts with different sphere radii and perturbations
    for restart in range(15):
        if time.time() > deadline:
            break
            
        if restart < 8:
            # Vary sphere radius
            r_sphere = 0.30 + 0.05 * (restart % 4)
            centers = _fibonacci_sphere(N, radius=r_sphere)
            centers += rng.normal(0, 0.005, centers.shape)
        elif restart < 12:
            # Fibonacci sphere with rotation
            centers = _fibonacci_sphere(N, radius=0.33)
            # Apply random rotation
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(-np.pi/4, np.pi/4)
            rot_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
            rot_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            centers = centers @ (rot_x @ rot_y).T
            centers += rng.normal(0, 0.005, centers.shape)
        else:
            # Mix: Fibonacci + small random perturbation
            centers = _fibonacci_sphere(N, radius=0.32)
            centers += rng.normal(0, 0.01, centers.shape)
        
        centers = np.clip(centers, -0.495, 0.495)
        
        # Run local search
        time_left = deadline - time.time()
        ls_time = min(6.0, time_left / (15 - restart))
        
        refined_centers, score = _local_search_sphere(centers, time_limit=ls_time)
        
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()
    
    # Final polish
    if best_centers is not None and (deadline - time.time()) > 10:
        final_time = deadline - time.time() - 5
        refined_centers, score = _local_search_sphere(best_centers, time_limit=final_time)
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()

    if best_centers is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
