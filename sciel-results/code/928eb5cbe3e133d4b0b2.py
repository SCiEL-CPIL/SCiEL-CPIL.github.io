"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _min_triangle_area_vec(points):
    """
    Vectorized computation of minimum triangle area.
    points: array of shape (n, 2)
    Returns the minimum area over all C(n,3) triangles.
    """
    n = len(points)
    # Compute all pairwise differences
    # For each triple (i, j, k), area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # Use broadcasting: shape (n, n, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 2)
    # cross product of diff[i,j] and diff[i,k] for all i,j,k
    # cross = diff[i,j,0]*diff[i,k,1] - diff[i,j,1]*diff[i,k,0]
    # We want i < j < k
    # Build area matrix: area[i,j,k] = 0.5*|cross(p_j-p_i, p_k-p_i)|
    # Using: cross(p_j-p_i, p_k-p_i) = cross(p_j, p_k) - cross(p_j, p_i) - cross(p_i, p_k) + cross(p_i, p_j)
    # But simpler: just compute for all triples via broadcasting
    # diff[i,j] = p_j - p_i
    # area for (i,j,k) = 0.5 * |diff[i,j,0]*diff[i,k,1] - diff[i,j,1]*diff[i,k,0]|
    cross = diff[:, :, 0][:, :, np.newaxis] * diff[:, np.newaxis, :, 1] - \
            diff[:, :, 1][:, :, np.newaxis] * diff[:, np.newaxis, :, 0]
    # cross[i,j,k] = cross(p_j - p_i, p_k - p_i)
    # We need i < j < k
    # Set upper triangle to inf
    areas = 0.5 * np.abs(cross)
    # Mask: only keep i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    areas[~mask] = np.inf
    return np.min(areas)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-10
    
    y = max(eps, y)
    y = min(y, h - eps)
    
    x_min_left = y / np.sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = max(x_min_left + eps, x)
    x = min(x_max_right - eps, x)
    
    return np.array([x, y])

def _clamp_all(points):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])

def _evaluate(points):
    """Evaluate the minimum triangle area for a set of points."""
    return _min_triangle_area_vec(points)

def _local_optimize(points, max_iter=3000, deadline=None):
    """
    Perform local optimization using Nelder-Mead.
    Returns (optimized_points, score).
    """
    x0 = np.array(points).flatten()
    
    def objective(x_flat):
        pts = np.array(x_flat).reshape(-1, 2)
        pts = _clamp_all(pts)
        return -_min_triangle_area_vec(pts)
    
    # Check deadline inside objective
    def objective_with_deadline(x_flat):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return objective(x_flat)
    
    result = minimize(
        objective_with_deadline,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
    )
    
    opt_points = np.array(result.x).reshape(-1, 2)
    opt_points = _clamp_all(opt_points)
    return opt_points, _evaluate(opt_points)

def _generate_initial_points(seed=None):
    """Generate diverse initial point configurations."""
    h = np.sqrt(3) / 2
    s3 = np.sqrt(3)
    
    if seed is not None:
        np.random.seed(seed)
    
    configs = []
    
    # Config 1: Symmetric hexagonal-like arrangement
    # Vertices of equilateral triangle + centroid + 6 surrounding points
    config1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.15, h/4],
        [0.85, h/4],
        [0.35, 3*h/8],
        [0.65, 3*h/8],
    ])
    configs.append(config1)
    
    # Config 2: Evenly spaced on a triangular grid
    # 3 vertices + 3 edge midpoints + 3 interior + 2 more
    config2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, 0.0],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.5, h/2],
    ])
    configs.append(config2)
    
    # Config 3: Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    config3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [inv_phi, h * inv_phi * 0.5],
        [1 - inv_phi, h * inv_phi * 0.5],
        [0.5, h * inv_phi],
        [0.25, h * 0.2],
        [0.75, h * 0.2],
        [0.5, h * 0.6],
        [0.3, h * 0.35],
        [0.7, h * 0.35],
    ])
    configs.append(config3)
    
    # Config 4: Perturbed symmetric
    base4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.15, h/4],
        [0.85, h/4],
        [0.35, 3*h/8],
        [0.65, 3*h/8],
    ])
    config4 = base4 + 0.03 * np.random.randn(11, 2)
    configs.append(config4)
    
    # Config 5: Random
    config5 = []
    for _ in range(N):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = h * v
        config5.append([x, y])
    configs.append(np.array(config5))
    
    # Config 6: Another symmetric pattern - 3 vertices + 3 edge points + 5 interior
    config6 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, h/3],
        [0.5, 2*h/3],
        [0.33, h/3],
        [0.67, h/3],
    ])
    configs.append(config6)
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    h = np.sqrt(3) / 2
    
    # Generate initial configurations
    initial_configs = _generate_initial_points(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Optimize each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 25:
            break
        
        points = _clamp_all(config)
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points.copy()
            best_score = current_score
        
        # Local optimization
        if time.time() < deadline - 15:
            opt_points, opt_score = _local_optimize(points, max_iter=2000, deadline=deadline - 10)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
    
    # Try additional random restarts if time permits
    rng = np.random.RandomState(123)
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate structured random points
        if restart % 3 == 0:
            # Symmetric perturbation
            base = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, h],
                [0.5, h/3],
                [0.25, h/6],
                [0.75, h/6],
                [0.5, 2*h/3],
                [0.15, h/4],
                [0.85, h/4],
                [0.35, 3*h/8],
                [0.65, 3*h/8],
            ])
            points = base + 0.04 * rng.randn(11, 2)
        elif restart % 3 == 1:
            # Grid-based perturbation
            base = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, h],
                [0.5, 0.0],
                [0.25, h/2],
                [0.75, h/2],
                [0.5, h/3],
                [0.25, h/6],
                [0.75, h/6],
                [0.5, 2*h/3],
                [0.5, h/2],
            ])
            points = base + 0.04 * rng.randn(11, 2)
        else:
            # Pure random
            points = []
            for _ in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u + 0.5 * v
                y = h * v
                points.append([x, y])
            points = np.array(points)
        
        points = _clamp_all(points)
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points.copy()
            best_score = current_score
        
        if time.time() < deadline - 12:
            opt_points, opt_score = _local_optimize(points, max_iter=1500, deadline=deadline - 8)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
    
    # Final check: ensure all points are inside the triangle
    if best_points is None:
        best_points = _clamp_all(initial_configs[0])
    else:
        best_points = _clamp_all(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
