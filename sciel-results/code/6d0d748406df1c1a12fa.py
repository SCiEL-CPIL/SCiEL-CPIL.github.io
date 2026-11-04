"""Heilbronn convex regions: place 13 points to maximize minimum triangle area / convex hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _triangle_areas_vectorized(points):
    """Compute all C(n,3) triangle areas vectorized."""
    n = len(points)
    # Get all index triples
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    # Cross product: 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = points[j] - points[i]
    v2 = points[k] - points[i]
    crosses = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(crosses)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_tri = np.min(areas)
    return min_tri / hull_area

def _smooth_min_objective(params, beta=50.0):
    """
    Smooth-min objective using ring parameterization.
    params: array of shape (n_rings, 2) where each row is [radius, angle_offset]
    Returns negative of smoothed minimum ratio.
    """
    n_rings = len(params)
    points = np.zeros((N, 2))
    
    # Assign points to rings based on known optimal structure for N=13
    # Use a configuration inspired by known Heilbronn solutions
    ring_sizes = [1, 4, 8]  # 1 center, 4 on inner ring, 8 on outer ring
    ring_radii = params[:, 0]
    ring_angles = params[:, 1]
    
    idx = 0
    for r_idx, size in enumerate(ring_sizes):
        radius = ring_radii[r_idx]
        base_angle = ring_angles[r_idx]
        if size == 1:
            points[idx] = [0.0, 0.0]
            idx += 1
        else:
            angles = base_angle + 2 * np.pi * np.arange(size) / size
            points[idx:idx+size, 0] = radius * np.cos(angles)
            points[idx:idx+size, 1] = radius * np.sin(angles)
            idx += size
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    ratio = areas / hull_area
    
    # Smooth min: -log(sum(exp(-beta * ratio))) / beta
    min_ratio = -np.log(np.sum(np.exp(-beta * ratio))) / beta
    return -min_ratio

def _initial_params():
    """Generate initial ring parameters."""
    # 3 rings: center, inner (4 pts), outer (8 pts)
    # Start with reasonable radii
    params = np.array([
        [0.0, 0.0],   # center
        [0.4, 0.0],   # inner ring radius, angle offset
        [0.8, np.pi/8]  # outer ring radius, angle offset
    ])
    return params

def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    deadline = time.time() + 135  # Safety margin
    
    # Multiple initial configurations to try
    initial_candidates = [
        np.array([[0.0, 0.0], [0.35, 0.0], [0.75, np.pi/8]]),
        np.array([[0.0, 0.0], [0.40, 0.0], [0.80, np.pi/8]]),
        np.array([[0.0, 0.0], [0.45, 0.0], [0.85, np.pi/8]]),
        np.array([[0.0, 0.0], [0.30, np.pi/12], [0.70, np.pi/12]]),
        np.array([[0.0, 0.0], [0.50, 0.0], [0.90, 0.0]]),
    ]
    
    best_points = None
    best_score = -1
    
    for init_params in initial_candidates:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                _smooth_min_objective,
                init_params,
                method='L-BFGS-B',
                bounds=[(0.05, 1.0), (0, 2*np.pi)] * 3,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Reconstruct points from optimized params
            params = result.x
            ring_sizes = [1, 4, 8]
            points = np.zeros((N, 2))
            idx = 0
            for r_idx, size in enumerate(ring_sizes):
                radius = params[r_idx, 0]
                base_angle = params[r_idx, 1]
                if size == 1:
                    points[idx] = [0.0, 0.0]
                    idx += 1
                else:
                    angles = base_angle + 2 * np.pi * np.arange(size) / size
                    points[idx:idx+size, 0] = radius * np.cos(angles)
                    points[idx:idx+size, 1] = radius * np.sin(angles)
                    idx += size
            
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue
    
    # If ring approach failed, fall back to a simple regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
