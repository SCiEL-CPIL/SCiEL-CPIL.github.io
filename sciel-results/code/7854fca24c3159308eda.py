"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        # Area of triangle with vertices points[i], points[j], points[k]
        # Using cross product: 0.5 * |cross(pj-pi, pk-pi)|
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    # Use scipy's convex hull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if all points are collinear or degenerate
        return 1e-10

def _objective(x, n=N):
    """Objective function to minimize: negative of (min_triangle_area / hull_area)."""
    points = x.reshape(n, 2)
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1.0  # Large penalty
    ratio = min_area / hull_area
    return -ratio

def construct():
    """Construct 13 points in a planar convex region to maximize the minimum triangle area ratio."""
    # Initial guess: points on a circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Use differential evolution for global optimization
    bounds = [(-1, 1)] * (2 * N)
    
    # Normalize the objective to be bounded for differential evolution
    # We want to maximize min_area / hull_area, so minimize -that ratio
    
    result = differential_evolution(
        _objective,
        bounds=bounds,
        args=(),
        maxiter=50,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=True
    )
    
    points = result.x.reshape(N, 2)
    
    # Refine with local optimization
    # Use L-BFGS-B to polish
    def neg_ratio(x):
        pts = x.reshape(N, 2)
        ma = _min_triangle_area(pts)
        ha = _convex_hull_area(pts)
        if ha < 1e-10:
            return 1.0
        return -(ma / ha)
    
    res_local = minimize(
        neg_ratio,
        points.flatten(),
        method='L-BFGS-B',
        bounds=[(-10, 10)] * (2 * N),
        options={'maxiter': 1000}
    )
    
    final_points = res_local.x.reshape(N, 2)
    
    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(final_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        final_points = final_points * scale
    
    return final_points
# EVOLVE-BLOCK-END
