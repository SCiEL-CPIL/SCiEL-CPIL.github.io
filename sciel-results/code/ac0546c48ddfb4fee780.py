"""Voronoi-diagram-based circle packing optimizer.

Represents the packing as a 2D Voronoi diagram where each circle is inscribed
in its cell, and optimizes the vertex positions of the diagram using a smooth
objective that maximizes the minimum inradius of all cells.
"""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

N = 50


def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _compute_voronoi_inradii(centers):
    """
    Compute the inradius of each Voronoi cell.
    
    For each site, the inradius is the minimum distance from the site to
    any vertex of its Voronoi cell, clipped by the distance to the boundary
    of the square [-0.5, 0.5]^2.
    
    Returns an array of shape (N,) with the inradius of each cell.
    """
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback: use pairwise distances only
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists, axis=1) / 2.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists, axis=1)
        return np.minimum(min_pair, min_wall)
    
    inradii = np.zeros(N)
    for i in range(N):
        # Get the vertices of the Voronoi cell for site i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        # Distance to walls
        wall_dists = 0.5 - np.abs(centers[i])
        min_wall = np.min(wall_dists)
        
        # Distance to Voronoi vertices
        if len(region) == 0:
            # Unbounded region (shouldn't happen for interior points)
            inradii[i] = min_wall
            continue
        
        # Get vertex coordinates
        vertex_indices = [v for v in region if v != -1]
        if len(vertex_indices) == 0:
            inradii[i] = min_wall
            continue
        
        vertices = vor.vertices[vertex_indices]
        dists_to_vertices = np.linalg.norm(vertices - centers[i], axis=1)
        min_vertex_dist = np.min(dists_to_vertices)
        
        inradii[i] = min(min_wall, min_vertex_dist)
    
    return inradii


def _optimize_voronoi(start_centers, bounds, time_budget=15.0):
    """
    Optimize circle centers using Voronoi-based inradius objective.
    
    The objective is a soft-min of the per-cell inradii, which encourages
    all circles to have roughly equal clearance.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    def objective(params, k):
        centers = params.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Compute inradii for each Voronoi cell
        inradii = _compute_voronoi_inradii(centers)
        
        if len(inradii) == 0:
            return 0.0
            
        m = np.min(inradii)
        shifted = inradii - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min
    
    # First pass with moderate k
    start_time = time.time()
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            args=(300.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False,
                     'maxfun': 5000}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_budget * 0.7:
        return best_centers, best_score
    
    # Second pass with higher k for polishing
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            args=(1000.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False,
                     'maxfun': 3000}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
    
    # Third pass: local refinement with even higher k
    if time.time() - start_time < time_budget:
        try:
            result = minimize(
                objective,
                best_centers.reshape(-1),
                args=(3000.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False,
                         'maxfun': 2000}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
    
    return best_centers, best_score


def _refine_with_local_search(centers, bounds, time_budget=5.0):
    """
    Local refinement: try small random perturbations of the worst-performing
    circles to escape local minima.
    """
    best_centers = centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    start_time = time.time()
    np.random.seed(777)
    
    for iteration in range(50):
        if time.time() - start_time > time_budget:
            break
        
        # Find the circle with the smallest clearance
        inradii = _compute_voronoi_inradii(best_centers)
        worst_idx = np.argmin(inradii)
        
        # Try perturbing the worst circle
        for attempt in range(5):
            if time.time() - start_time > time_budget:
                break
            
            perturbation = np.random.normal(0, 0.005, (N, 2))
            perturbation[worst_idx] *= 3.0  # Larger perturbation for worst circle
            
            new_centers = best_centers + perturbation
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _calculate_true_radius(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
    
    return best_centers, best_score


def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    s = 0.14
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.02, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    time_budget = 110.0
    
    for i, init in enumerate(inits):
        remaining = time_budget - (time.time() - start_time)
        if remaining < 15.0:
            break
            
        per_init_budget = min(remaining / max(1, len(inits) - i), 20.0)
        
        centers, score = _optimize_voronoi(init, bounds, per_init_budget)
        
        # Add local refinement
        if time.time() - start_time < time_budget - 5.0:
            centers, score = _refine_with_local_search(centers, bounds, 3.0)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
