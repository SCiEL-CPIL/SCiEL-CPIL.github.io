"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay

def _compute_radius(centers):
    """Compute the packing radius for a set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.sqrt(np.min(dists_sq[iu]))
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _voronoi_lloyd_step(centers, step=1.0):
    """
    One Lloyd iteration: move each point toward the centroid of its Voronoi cell.
    This equalizes clearance, a key property of optimal packings.
    """
    try:
        vor = Voronoi(centers)
    except Exception:
        return centers.copy()
    
    new_centers = centers.copy()
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        vertices = vor.vertices[region]
        if len(vertices) == 0:
            continue
        # Centroid of the Voronoi cell
        centroid = np.mean(vertices, axis=0)
        # Move toward centroid, but clamp to cube
        move = centroid - centers[i]
        new_pos = centers[i] + step * move
        new_pos = np.clip(new_pos, -0.499, 0.499)
        new_centers[i] = new_pos
    
    return new_centers

def _lloyd_optimize(centers, time_limit=5.0, max_iter=200):
    """Run Lloyd iterations with adaptive step size."""
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_radius(centers)
    step = 0.5
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_centers = _voronoi_lloyd_step(centers, step)
        new_score = _compute_radius(new_centers)
        
        if new_score >= best_score:
            centers = new_centers
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
            step = min(step * 1.05, 1.0)
        else:
            step *= 0.7
            if step < 0.01:
                break
    
    return best_centers, best_score

def _softmin_polish(centers, time_limit=5.0):
    """
    Smooth optimization using a soft-min of the minimum distance.
    Gradient-based refinement of the packing.
    """
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_radius(centers)
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    # Anneal the soft-min parameter
    t = 0.05
    lr = 0.005
    
    while time.time() - start_time < time_limit:
        # Compute distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        pair_dists = np.sqrt(np.maximum(dists_sq[iu], 1e-12))
        
        # Wall distances per sphere
        wall_dists_per_sphere = np.min(0.5 - np.abs(centers), axis=1)
        
        # Soft-min of all constraints (pair dists/2 and wall dists)
        all_constraints = np.concatenate([pair_dists / 2.0, wall_dists_per_sphere])
        n_constraints = len(all_constraints)
        
        # Soft-min: -t * log(sum(exp(-c/t)))
        # Gradient w.r.t. centers
        weights = np.exp(-all_constraints / t)
        weights /= np.sum(weights)
        
        # Build gradient
        grad = np.zeros_like(centers)
        
        # Pair contributions
        pair_weights = weights[:n_pairs]
        pair_diffs = diff[iu[0], iu[1], :]
        pair_dists_safe = np.maximum(pair_dists, 1e-8)
        
        for k in range(n_pairs):
            w = pair_weights[k] / 2.0
            d = pair_diffs[k]
            dist = pair_dists_safe[k]
            # d/dx_i (|x_i - x_j|/2) = d / (2*dist)
            grad[iu[0][k]] += w * d / dist
            grad[iu[1][k]] -= w * d / dist
        
        # Wall contributions
        wall_weights = weights[n_pairs:]
        for i in range(N):
            w = wall_weights[i]
            # d/dx_i (0.5 - |x_i|) = -sign(x_i)
            grad[i] -= w * np.sign(centers[i])
        
        # Gradient ascent (we want to maximize the soft-min)
        centers = centers + lr * grad
        centers = np.clip(centers, -0.499, 0.499)
        
        current_score = _compute_radius(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
        
        # Anneal
        t *= 0.95
        lr *= 0.98
    
    return best_centers, best_score

def _local_search(centers, time_limit=3.0):
    """Direct local search on critical constraints."""
    start_time = time.time()
    centers = centers.copy()
    current_score = _compute_radius(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, k=1)
    step_size = 0.015
    
    for iteration in range(3000):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        pair_dists_sq = dists_sq[iu]
        min_pair_dist = np.sqrt(np.min(pair_dists_sq))
        
        wall_dists_per_sphere = np.min(0.5 - np.abs(centers), axis=1)
        global_min_wall = np.min(wall_dists_per_sphere)
        
        current_r = min(min_pair_dist / 2.0, global_min_wall)
        
        min_pair_idx = np.argmin(pair_dists_sq)
        i, j = iu[0][min_pair_idx], iu[1][min_pair_idx]
        min_wall_idx = np.argmin(wall_dists_per_sphere)
        
        candidates = list(set([i, j, min_wall_idx]))
        if np.random.rand() < 0.15:
            candidates.append(np.random.randint(N))
        
        improved = False
        for idx in candidates:
            for _ in range(8):
                direction = np.random.normal(0, 1, 3)
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                direction /= norm
                new_pos = centers[idx] + direction * step_size
                
                if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                    continue
                
                old_pos = centers[idx]
                centers[idx] = new_pos
                new_score = _compute_radius(centers)
                
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
                step_size = min(step_size, 0.08)
        
        if iteration % 100 == 0:
            step_size = max(step_size, 0.01)

    return best_centers, best_score

def _seed_fcc_like():
    """Generate an FCC-like lattice seed."""
    points = []
    # FCC lattice points in [0,1]^3, scaled to [-0.5, 0.5]^3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # FCC: corners + face centers
                # Corner
                points.append([i, j, k])
                # Face centers
                if i < 2:
                    points.append([i + 0.5, j, k])
                    points.append([i + 0.5, j + 0.5, k])
                    points.append([i + 0.5, j, k + 0.5])
                if j < 2:
                    points.append([i, j + 0.5, k])
                    points.append([i, j + 0.5, k + 0.5])
                if k < 2:
                    points.append([i, j, k + 0.5])
                    points.append([i + 0.5, j, k + 0.5])
    
    points = np.array(points)
    # Scale to fit in [-0.5, 0.5]
    points = points / 2.5 - 0.5
    # Filter to those inside the cube
    mask = np.all(points >= -0.49, axis=1) & np.all(points <= 0.49, axis=1)
    points = points[mask]
    
    # Select N points with good spread
    if len(points) >= N:
        # Greedy selection for maximum minimum distance
        selected = [np.random.choice(len(points))]
        for _ in range(N - 1):
            dists = np.min(np.linalg.norm(points[:, np.newaxis, :] - points[selected][np.newaxis, :, :], axis=2), axis=1)
            selected.append(np.argmax(dists))
        return points[selected]
    else:
        # Pad with random points
        extra = np.random.uniform(-0.45, 0.45, (N - len(points), 3))
        return np.vstack([points, extra])

def _seed_grid(rng):
    """Generate a grid-based seed with perturbation."""
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = rng.permutation(len(grid))[:N]
    centers = grid[idx].copy()
    centers += rng.normal(0, 0.02, centers.shape)
    return np.clip(centers, -0.49, 0.49)

def _seed_random(rng):
    """Generate a random seed."""
    return rng.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different seeds and optimization pipelines
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        # Choose seed strategy
        if restart == 0:
            centers = _seed_fcc_like()
        elif restart < 4:
            centers = _seed_grid(rng)
        else:
            centers = _seed_random(rng)
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Pipeline: Lloyd -> Soft-min polish -> Local search
        # Allocate time proportionally
        t_lloyd = min(4.0, time_left * 0.3)
        t_polish = min(4.0, time_left * 0.3)
        t_ls = min(4.0, time_left * 0.3)
        
        # Stage 1: Lloyd equalization
        if time.time() < deadline - 10:
            centers, score1 = _lloyd_optimize(centers, time_limit=t_lloyd)
        
        # Stage 2: Soft-min gradient polish
        if time.time() < deadline - 8:
            centers, score2 = _softmin_polish(centers, time_limit=t_polish)
        
        # Stage 3: Local search
        if time.time() < deadline - 5:
            centers, score3 = _local_search(centers, time_limit=t_ls)
        
        if score3 > best_score:
            best_score = score3
            best_centers = centers.copy()
    
    # Final polish on the best candidate
    if best_centers is not None and (deadline - time.time()) > 8:
        final_time = deadline - time.time() - 3
        # Run a final local search with more time
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
