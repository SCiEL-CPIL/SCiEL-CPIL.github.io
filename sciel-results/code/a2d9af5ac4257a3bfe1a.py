"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def _min_triangle_area_normalized(points):
    """
    Compute the minimum area of any triangle formed by 3 points,
    normalized by the convex hull area.
    
    points: (N, 2) array
    Returns: float, the normalized min triangle area (higher is better)
    """
    n = len(points)
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if hull fails, use a large penalty
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
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _objective_gaps(x):
    """
    Objective function for optimization using gap parameters.
    x: (N,) array of log-gap parameters.
    Returns: float, negative of the normalized min triangle area.
    """
    # Convert log-gaps to actual gaps that sum to 2*pi
    # exp(x) ensures positivity
    # Normalize to sum to 2*pi
    exp_x = np.exp(x - np.max(x))  # Stability
    gaps = exp_x / np.sum(exp_x) * 2 * np.pi
    
    # Calculate angles (cumulative sum)
    # Points are at 0, g0, g0+g1, ...
    angles = np.zeros(N)
    for i in range(1, N):
        angles[i] = np.sum(gaps[:i])
        
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    score = _min_triangle_area_normalized(points)
    return -score


def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    
    Returns: (N, 2) array of points
    """
    # Bounds for x
    # Regular polygon corresponds to x=0 (all gaps equal).
    # We allow deviations to explore non-uniform spacings.
    bounds = [(-5, 5) for _ in range(N)]
    
    # Use differential_evolution for global optimization
    # This is robust for this type of problem and handles the non-convexity well
    result = differential_evolution(
        _objective_gaps,
        bounds=bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        popsize=15
    )
    
    optimal_x = result.x
    exp_x = np.exp(optimal_x - np.max(optimal_x))
    gaps = exp_x / np.sum(exp_x) * 2 * np.pi
    
    angles = np.zeros(N)
    for i in range(1, N):
        angles[i] = np.sum(gaps[:i])
        
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
# EVOLVE-BLOCK-END
