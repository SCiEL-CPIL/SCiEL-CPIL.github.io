"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 115.0  # safety margin before 120s limit


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
                        (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: compute area using shoelace formula on convex hull
        from scipy.spatial import ConvexHull as CH
        try:
            hull = CH(points)
            return hull.area if hasattr(hull, 'area') else hull.volume
        except Exception:
            # Last resort: use all points in order
            return 1.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area <= 0 or not np.isfinite(hull_area):
        return 0.0
    return min_area / hull_area


def _soft_min_triangle_area(points, alpha=50.0):
    """Compute a soft approximation of the minimum triangle area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                cross = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
                        (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                areas.append(0.5 * abs(cross))
    areas = np.array(areas)
    # Soft min using log-sum-exp: soft_min = -log(sum(exp(-alpha * x))) / alpha
    # This is a smooth approximation of min(x)
    scaled = -alpha * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:  # avoid overflow
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    else:
        log_sum = np.log(np.sum(np.exp(scaled)))
    return -log_sum / alpha


def _negative_soft_score(points, alpha=50.0):
    """Negative of soft score for minimization."""
    soft_min_area = _soft_min_triangle_area(points, alpha)
    hull_area = _hull_area(points)
    if hull_area <= 0 or not np.isfinite(hull_area):
        return 1e10
    return -soft_min_area / hull_area


def _negative_score(points):
    """Negative of score for minimization."""
    return -_score(points)


def _initial_points(seed=42):
    """Generate initial points: regular polygon with some perturbation."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(N) / N + rng.uniform(0, 0.1, size=N)
    radii = 1.0 + rng.normal(0, 0.05, size=N)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    return points


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initializations and optimization runs
    n_trials = 8
    trial_idx = 0
    
    while trial_idx < n_trials:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        seed = 42 + trial_idx * 100
        points = _initial_points(seed)
        
        # Optimize using Powell method
        try:
            result = minimize(
                _negative_score,
                points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 20000}
            )
            
            opt_points = result.x.reshape(-1, 2)
            score = _score(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
        except Exception:
            pass
        
        # Also try with soft-min surrogate
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        try:
            result2 = minimize(
                _negative_soft_score,
                points.flatten(),
                method='Powell',
                options={'maxiter': 300, 'maxfev': 15000}
            )
            
            opt_points2 = result2.x.reshape(-1, 2)
            score2 = _score(opt_points2)
            
            if score2 > best_score:
                best_score = score2
                best_points = opt_points2.copy()
        except Exception:
            pass
        
        trial_idx += 1
    
    # If no valid points found, return a default
    if best_points is None:
        best_points = _initial_points(42)
    
    # Ensure we have exactly N points
    if len(best_points) != N:
        best_points = _initial_points(42)
    
    return best_points


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
