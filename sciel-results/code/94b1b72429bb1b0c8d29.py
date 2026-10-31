"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import differential_evolution

N = 13

def _min_triangle_area(points):
    """Compute minimum triangle area."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute score: min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _normalize(points):
    """Normalize points to unit area convex hull."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return points
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = 1.0 / np.sqrt(hull_area)
    return points * scale

def _generate_points_from_params(params):
    """
    Generate 13 points from parameters.
    Params:
    - 13 angular deviations (delta_i) from regular 13-gon
    - 2 scaling factors (sx, sy)
    - 1 rotation angle (phi)
    """
    n = 13
    deltas = params[:n]
    sx = params[n]
    sy = params[n+1]
    phi = params[n+2]
    
    # Base angles for regular 13-gon
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = base_angles + deltas
    
    # Points on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    # Apply scaling
    points[:, 0] *= sx
    points[:, 1] *= sy
    
    # Apply rotation
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi, cos_phi]])
    
    points = points @ R.T
    
    return points

def _objective(params):
    """Objective function for Differential Evolution (minimize negative score)."""
    points = _generate_points_from_params(params)
    score = _score(points)
    if score == 0.0:
        return 1.0 # Penalize degenerate cases
    return -score

def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 13
    # Define bounds for parameters
    # Angles: +/- pi/3 (large enough to explore, small enough to keep convexity likely)
    angle_bounds = [(-np.pi/3, np.pi/3) for _ in range(n)]
    # Scaling: 0.1 to 5.0
    scale_bounds = [(0.1, 5.0), (0.1, 5.0)]
    # Rotation: 0 to pi/2 (symmetry allows reducing range)
    rotation_bounds = [(0, np.pi/2)]
    
    bounds = angle_bounds + scale_bounds + rotation_bounds
    
    # Initial population size
    popsize = 15
    
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=popsize,
            tol=1e-6,
            polish=True,
            init='latinhypercube'
        )
        
        best_params = result.x
        best_score_val = -result.fun
        
    except Exception:
        # Fallback if DE fails
        best_params = np.zeros(16)
        best_score_val = 0.0

    best_points = _generate_points_from_params(best_params)
    best_points = _normalize(best_points)
    
    # Final check
    final_score = _score(best_points)
    
    return best_points
