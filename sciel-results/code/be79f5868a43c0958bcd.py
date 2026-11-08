"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import itertools


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the point set."""
    # Sort points by angle around centroid for simplicity, but better to use a proper hull.
    # For a strictly convex set of points in order, we can use the shoelace formula.
    # However, we need the actual convex hull. Let's use a simple approach:
    # Since our points are generated in angular order around the center and are strictly convex,
    # the convex hull is just the points themselves in order.
    # But to be safe, let's compute it properly.
    
    # Sort by x, then y
    pts = np.array(points)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenate
    hull = lower[:-1] + upper[:-1]
    hull = np.array(hull)
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def _is_strictly_convex(points):
    """Check if points form a strictly convex polygon (in given order)."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross <= 1e-10:  # Allow small tolerance
            return False
    return True


def _score_from_params(params):
    """
    Convert perturbation parameters to points and compute score.
    params: array of shape (2*N,)
      - params[0:N] are angular perturbations
      - params[N:2*N] are radial perturbations
    """
    base_angles = 2 * np.pi * np.arange(N) / N
    base_radius = 1.0
    
    # Perturb angles and radii
    angles = base_angles + params[0:N]
    radii = base_radius + params[N:2*N]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.1)
    
    # Convert to Cartesian
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Sort by angle to ensure convex order
    # The angles should already be in order if perturbations are small
    # But let's sort to be safe
    sorted_idx = np.argsort(angles)
    points = points[sorted_idx]
    
    # Check strict convexity
    if not _is_strictly_convex(points):
        return 0.0
    
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    
    if hull_area <= 1e-10:
        return 0.0
    
    return min_area / hull_area


def construct():
    """
    Construct 14 points in a planar convex region normalized to unit area
    to maximize the minimum triangle area.
    
    Returns:
        np.array of shape (14, 2)
    """
    # Parameter bounds for angular and radial perturbations
    # Angular perturbations should be small to maintain convexity
    # The base angle spacing is 2*pi/14 ~ 0.4488
    # Allow perturbations up to ~0.3 of the spacing to maintain order
    max_angle_pert = 0.3 * (2 * np.pi / N)
    max_radius_pert = 0.5  # Allow radii to vary significantly
    
    bounds = [(-max_angle_pert, max_angle_pert)] * N + [(-max_radius_pert, max_radius_pert)] * N
    
    # Use differential evolution to maximize the score
    # We'll minimize the negative score
    def objective(params):
        return -_score_from_params(params)
    
    # Run differential evolution with a reasonable number of iterations
    # Time limit is 120s, so we need to be careful
    result = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=True
    )
    
    best_params = result.x
    points = np.column_stack([
        (1.0 + best_params[N:2*N]) * np.cos(2 * np.pi * np.arange(N) / N + best_params[0:N]),
        (1.0 + best_params[N:2*N]) * np.sin(2 * np.pi * np.arange(N) / N + best_params[0:N])
    ])
    
    # Sort by angle
    angles = 2 * np.pi * np.arange(N) / N + best_params[0:N]
    sorted_idx = np.argsort(angles)
    points = points[sorted_idx]
    
    # Normalize to unit area by scaling
    hull_area = _convex_hull_area(points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        points = points * scale
    
    return points
# EVOLVE-BLOCK-END
