"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _triangle_areas(points):
    """Compute all C(n,3) triangle areas for n points."""
    n = len(points)
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        # Area of triangle = 0.5 * |cross product|
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        areas.append(area)
    return np.array(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on convex hull points
        from scipy.spatial import ConvexHull as CH
        try:
            hull = CH(points)
            return hull.volume
        except Exception:
            # If all points are collinear or degenerate, return small value
            return 1e-10

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _soft_min_triangle_area(points, beta=10.0):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    We want to maximize the min area, so we minimize the soft-min.
    soft_min(areas) = -1/beta * log(sum(exp(-beta * area)))
    This is a smooth lower bound on the min area.
    """
    areas = _triangle_areas(points)
    # Clip to avoid numerical issues
    areas = np.clip(areas, 1e-12, None)
    # Log-sum-exp for soft minimum
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum / beta
    return soft_min

def _smooth_objective(points, beta=10.0):
    """
    Smooth objective: minimize (soft_min_triangle_area / convex_hull_area)
    But we want to maximize min_area / hull_area, so we minimize the negative.
    Actually, let's minimize the soft_min / hull_area, which approximates min_area/hull_area.
    To maximize this, we minimize the negative.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    soft_min = _soft_min_triangle_area(points, beta)
    return soft_min / hull_area

def _generate_initializations():
    """Generate diverse structured initializations."""
    inits = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(poly)
    
    # Concentric rings
    # Ring 1: 7 points on outer circle
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    # Ring 2: 7 points on inner circle
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    ring2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    inits.append(np.vstack([ring1, ring2]))
    
    # Another ring configuration
    angles1b = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1b = np.column_stack([np.cos(angles1b), np.sin(angles1b)])
    angles2b = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring2b = 0.6 * np.column_stack([np.cos(angles2b), np.sin(angles2b)])
    inits.append(np.vstack([ring1b, ring2b]))
    
    # Square-like with interior points
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float)
    interior = np.zeros((10, 2))
    for i in range(10):
        interior[i, 0] = 0.5 * np.cos(2 * np.pi * i / 10)
        interior[i, 1] = 0.5 * np.sin(2 * np.pi * i / 10)
    inits.append(np.vstack([corners, interior]))
    
    # Random but seeded
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.randn(N, 2)
        # Normalize to unit area roughly
        hull_area = _convex_hull_area(pts)
        if hull_area > 1e-10:
            pts = pts / np.sqrt(hull_area)
        inits.append(pts)
    
    return inits

def construct():
    global _start_time
    best_points = None
    best_score = -1e10
    
    # Try multiple initializations
    inits = _generate_initializations()
    
    for init in inits:
        if time.time() > _DEADLINE - 10:
            break
        
        # Optimize using L-BFGS-B on the smooth objective
        try:
            result = minimize(
                _smooth_objective,
                init,
                method='L-BFGS-B',
                args=(10.0,),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < 1e10:
                score = -result.fun  # Since we minimized soft_min/hull_area
                # Actually, we need to compute the actual score
                actual_score = _score(result.x)
                
                if actual_score > best_score:
                    best_score = actual_score
                    best_points = result.x.copy()
                    
        except Exception:
            continue
        
        # Also try Powell on the raw objective for refinement
        if time.time() > _DEADLINE - 15:
            continue
            
        try:
            result2 = minimize(
                _negative_score,
                result.x if result.success else init,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            
            if result2.success or result2.fun < 0:
                actual_score2 = -result2.fun
                if actual_score2 > best_score:
                    best_score = actual_score2
                    best_points = result2.x.copy()
                    
        except Exception:
            continue
    
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
