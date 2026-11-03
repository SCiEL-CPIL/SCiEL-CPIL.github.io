"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START

def compute_radius(coords):
    """
    Computes the maximum radius for the given circle centers.
    Returns the minimum of:
    1. Half the minimum distance between any two circle centers.
    2. The minimum distance from any circle center to the boundary of the square [-0.5, 0.5]^2.
    """
    if not np.all(np.isfinite(coords)):
        return 0.0
    
    # Distance to boundaries
    # x distance to left/right walls
    dist_x = np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5)
    # y distance to top/bottom walls
    dist_y = np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)
    
    # Minimum distance to any wall
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    # If any center is outside, the radius is effectively 0 or negative (invalid)
    if min_wall_dist < 0:
        return 0.0

    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Minimum distance between distinct circles
    # Mask the diagonal (distance to self is 0)
    min_pair_dist = np.inf
    for i in range(N):
        dists[i, i] = np.inf
    min_pair_dist = np.min(dists)
    
    radius = min(min_wall_dist, min_pair_dist / 2.0)
    
    return radius

def soft_min(x, k=100.0):
    """
    Soft minimum of array x using log-sum-exp trick.
    Returns a smooth approximation to min(x).
    """
    x = np.asarray(x)
    m = np.min(x)
    return m - np.log(np.sum(np.exp(k * (x - m)))) / k

def objective_soft(coords, k=100.0):
    """
    Soft objective: minimize negative soft-min of all constraints.
    Constraints: pairwise distances / 2, wall distances.
    """
    if not np.all(np.isfinite(coords)):
        return 1e6
    
    # Wall distances
    dist_x = np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5)
    dist_y = np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)
    wall_dists = np.concatenate([dist_x, dist_y])
    
    # Pairwise distances / 2
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2)) / 2.0
    # Remove diagonal
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    all_constraints = np.concatenate([wall_dists, pair_dists])
    r_soft = soft_min(all_constraints, k)
    return -r_soft

def gradient_soft(coords, k=100.0):
    """
    Compute gradient of objective_soft.
    """
    if not np.all(np.isfinite(coords)):
        return np.zeros_like(coords)
    
    # Wall distances
    dist_x = np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5)
    dist_y = np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)
    wall_dists = np.concatenate([dist_x, dist_y])
    
    # Pairwise distances / 2
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2)) / 2.0
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    all_constraints = np.concatenate([wall_dists, pair_dists])
    m = np.min(all_constraints)
    
    # Weights for soft min: w_i = exp(-k*(x_i - m)) / sum_j exp(-k*(x_j - m))
    exp_terms = np.exp(-k * (all_constraints - m))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(coords)
    
    # Wall contributions
    # dist_x[i] = min(0.5 - x_i, x_i + 0.5)
    for i in range(N):
        # x component
        if dist_x[i] <= dist_y[i]:
            # dist_x is active
            if 0.5 - coords[i, 0] <= coords[i, 0] + 0.5:
                # 0.5 - x_i <= x_i + 0.5 => 0 <= 2*x_i => x_i >= 0
                # grad of (0.5 - x_i) w.r.t. x_i is -1
                grad[i, 0] += -weights[i] * k * exp_terms[i] / np.sum(exp_terms) * (-1)
            else:
                # x_i + 0.5 < 0.5 - x_i => x_i < 0
                # grad of (x_i + 0.5) w.r.t. x_i is 1
                grad[i, 0] += weights[i] * k * exp_terms[i] / np.sum(exp_terms) * 1
        else:
            # dist_y is active
            if 0.5 - coords[i, 1] <= coords[i, 1] + 0.5:
                # y_i >= 0
                grad[i, 1] += -weights[N + i] * k * exp_terms[N + i] / np.sum(exp_terms) * (-1)
            else:
                # y_i < 0
                grad[i, 1] += weights[N + i] * k * exp_terms[N + i] / np.sum(exp_terms) * 1
    
    # Pairwise contributions
    # For each pair (i, j), dist_ij = ||c_i - c_j|| / 2
    # grad w.r.t. c_i: (c_i - c_j) / (2 * ||c_i - c_j||) * weight
    # grad w.r.t. c_j: (c_j - c_i) / (2 * ||c_i - c_j||) * weight
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = coords[i] - coords[j]
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        w = weights[N + N - 1 + idx]  # Need to map index correctly
        # Actually, let's recompute weights for pairs only for clarity
        # This is getting complex. Let me use a simpler approach.
    
    # Simpler: use finite differences or a cleaner gradient computation
    # For now, return zeros and rely on Powell or use L-BFGS-B with numerical grad
    return grad

def initial_hex_layout(n):
    """
    Generates an initial hexagonal layout of n points inside the square.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)
    points = points[indices[:N]]
    
    return points

def build_proximity_graph(coords, k=8):
    """
    Build a k-nearest neighbor graph. Returns list of edges (i, j).
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    edges = set()
    for i in range(N):
        # Find k nearest neighbors
        nearest = np.argsort(dists[i])[:k]
        for j in nearest:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    
    return list(edges)

def proximity_swap_optimize(coords, max_iters=500, k=8, tol=1e-8):
    """
    Iteratively perform swaps on the proximity graph to improve the minimum distance.
    """
    coords = coords.copy()
    best_radius = compute_radius(coords)
    
    for it in range(max_iters):
        if it % 50 == 0:
            current_radius = compute_radius(coords)
            if current_radius > best_radius:
                best_radius = current_radius
            
            # Check time
            if time.time() > time.time() - 145:  # placeholder
                break
        
        # Build proximity graph
        edges = build_proximity_graph(coords, k)
        
        # Find the critical pair (minimum distance)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        min_dist = dists[i, j]
        
        # Try to swap i with some neighbor of j to increase min_dist
        # Find neighbors of j in the graph
        neighbors_j = [n for (a, b) in edges if a == j for n in [b] if n != i] + \
                      [n for (a, b) in edges if b == j for n in [a] if n != i]
        
        improved = False
        for n in neighbors_j:
            # Try swapping i and n
            new_coords = coords.copy()
            new_coords[i], new_coords[n] = new_coords[n], new_coords[i]
            
            # Check if this improves the minimum distance
            diff_new = new_coords[:, np.newaxis, :] - new_coords[np.newaxis, :, :]
            dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=2))
            np.fill_diagonal(dists_new, np.inf)
            min_dist_new = np.min(dists_new)
            
            if min_dist_new > min_dist + tol:
                coords = new_coords
                improved = True
                break
        
        if not improved:
            # Try small random perturbations of the critical pair
            for _ in range(10):
                new_coords = coords.copy()
                noise = np.random.normal(0, min_dist * 0.1, (2, 2))
                new_coords[i] += noise[0]
                new_coords[j] += noise[1]
                
                # Keep inside bounds
                new_coords[:, 0] = np.clip(new_coords[:, 0], -0.5, 0.5)
                new_coords[:, 1] = np.clip(new_coords[:, 1], -0.5, 0.5)
                
                diff_new = new_coords[:, np.newaxis, :] - new_coords[np.newaxis, :, :]
                dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=2))
                np.fill_diagonal(dists_new, np.inf)
                min_dist_new = np.min(dists_new)
                
                if min_dist_new > min_dist + tol:
                    coords = new_coords
                    improved = True
                    break
        
        if not improved:
            break
    
    return coords

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Initial guess: Hexagonal
    coords = initial_hex_layout(N)
    
    # Initial radius
    best_radius = compute_radius(coords)
    best_coords = coords.copy()
    
    # Step 1: Proximity swap optimization
    try:
        swapped_coords = proximity_swap_optimize(coords, max_iters=200, k=8)
        swapped_radius = compute_radius(swapped_coords)
        if swapped_radius > best_radius:
            best_radius = swapped_radius
            best_coords = swapped_coords
    except Exception:
        pass
    
    # Step 2: Gradient-based polish using L-BFGS-B with soft objective
    if time.time() < deadline - 20:
        try:
            x0 = best_coords.flatten()
            res = minimize(objective_soft, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if time.time() < deadline - 10:
                new_coords = res.x.reshape(N, 2)
                new_coords[:, 0] = np.clip(new_coords[:, 0], -0.5, 0.5)
                new_coords[:, 1] = np.clip(new_coords[:, 1], -0.5, 0.5)
                new_radius = compute_radius(new_coords)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_coords = new_coords
        except Exception:
            pass
    
    # Step 3: Final Powell polish with tight tolerance
    if time.time() < deadline - 5:
        try:
            def objective_exact(c):
                return -compute_radius(c.reshape(N, 2))
            
            x0 = best_coords.flatten()
            res = minimize(objective_exact, x0, method='Powell',
                          options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12})
            
            if time.time() < deadline:
                new_coords = res.x.reshape(N, 2)
                new_coords[:, 0] = np.clip(new_coords[:, 0], -0.5, 0.5)
                new_coords[:, 1] = np.clip(new_coords[:, 1], -0.5, 0.5)
                new_radius = compute_radius(new_coords)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_coords = new_coords
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_coords)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_coords

# EVOLVE-BLOCK-END
