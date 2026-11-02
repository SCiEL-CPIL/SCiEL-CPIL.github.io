"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area_normalized(points):
    """
    Compute the minimum area of any triangle formed by 3 points,
    normalized by the convex hull area.
    
    points: (N, 2) array
    Returns: float, the normalized min triangle area (higher is better)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for near-duplicate points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Find minimum triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _objective(params):
    """
    Objective function for optimization: minimize negative of min normalized triangle area.
    
    params: (N,) array of angles (radians) on a circle
    Returns: float, negative of the normalized min triangle area
    """
    angles = params
    # Sort angles to ensure convex position ordering
    angles_sorted = np.sort(angles)
    points = np.column_stack([np.cos(angles_sorted), np.sin(angles_sorted)])
    
    score = _min_triangle_area_normalized(points)
    return -score


def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    
    Returns: (N, 2) array of points
    """
    # Start with equally spaced angles (regular polygon)
    initial_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Try multiple restarts with perturbations to escape local optima
    best_score = -np.inf
    best_angles = initial_angles.copy()
    
    rng = np.random.default_rng(42)
    
    for restart in range(15):
        if restart == 0:
            angles = initial_angles.copy()
        else:
            # Random perturbation scaled by restart number
            scale = 0.05 + 0.01 * restart
            angles = initial_angles + rng.normal(0, scale, N)
        
        # Optimize using Nelder-Mead
        result = minimize(
            _objective,
            angles,
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        optimal_angles = result.x
        score = -result.fun
        
        if score > best_score:
            best_score = score
            best_angles = optimal_angles.copy()
    
    # Sort angles to get final points in convex position
    best_angles_sorted = np.sort(best_angles)
    points = np.column_stack([np.cos(best_angles_sorted), np.sin(best_angles_sorted)])
    
    return points
# EVOLVE-BLOCK-END
