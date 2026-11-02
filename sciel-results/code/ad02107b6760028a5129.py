import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
MAX_TIME = 115.0  # Leave a safety margin for the evaluator's 120s limit

def _hull_area(points):
    """Calculate the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Calculate the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    # Optimization: Only check triples. O(n^3) is fine for n=14 (14*13*12/6 = 364 triples)
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Calculate the score: min triangle area / hull area."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    ha = _hull_area(points)
    if ha < 1e-9:
        return 0.0
        
    ma = _min_triangle_area(points)
    if ma < 1e-9:
        return 0.0
        
    return ma / ha

def _regular_polygon(k, r=1.0, phase=0.0):
    """Generate a regular k-gon."""
    a = phase + 2 * np.pi * np.arange(k) / k
    return r * np.column_stack([np.cos(a), np.sin(a)])

def _fourier_to_points(coeffs, n_points=N):
    """
    Convert Fourier coefficients to points on the curve.
    coeffs: [a1, b1, a2, b2, ..., aK, bK]
    r(theta) = 1 + sum (a_k cos(k theta) + b_k sin(k theta))
    Points are placed at uniform angular intervals.
    """
    K = len(coeffs) // 2
    if K == 0:
        return _regular_polygon(n_points)
        
    theta = 2 * np.pi * np.arange(n_points) / n_points
    
    # Compute radius for each point
    r = np.ones(n_points)
    for k in range(1, K + 1):
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
        r += a * np.cos(k * theta) + b * np.sin(k * theta)
        
    # Check for convexity violation (r must be positive)
    # A stricter check is r + r'' > 0, but for small coeffs, r > 0 is a good heuristic.
    # If r becomes <= 0, the curve self-intersects or is invalid.
    if np.any(r <= 0.1): # Small margin
        return None
        
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _objective(coeffs):
    """
    Objective function for minimization.
    We want to maximize (min_tri_area / hull_area), so we minimize its reciprocal.
    If the shape is invalid, return a large penalty.
    """
    pts = _fourier_to_points(coeffs)
    if pts is None:
        return 1e6
        
    # Check if the polygon is convex. 
    # The Fourier curve with small coeffs is convex, but let's verify the points form a convex hull.
    # Actually, if r(theta) > 0 and r(theta) + r''(theta) > 0, it's convex.
    # For optimization speed, we trust the Fourier expansion for small coeffs.
    
    ha = _hull_area(pts)
    if ha < 1e-6:
        return 1e6
        
    ma = _min_triangle_area(pts)
    if ma < 1e-6:
        return 1e6
        
    ratio = ma / ha
    if ratio <= 0:
        return 1e6
        
    return 1.0 / ratio

def _optimize_shape(start_coeffs, deadline, maxiter=1000):
    """Run Nelder-Mead optimization on the Fourier coefficients."""
    if time.time() > deadline:
        return None, None
        
    try:
        result = minimize(
            _objective, 
            start_coeffs, 
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        return result.x, result.fun
    except Exception:
        return None, None

def _normalize_unit_area(points):
    """Scale points so that the convex hull area is 1."""
    ha = _hull_area(points)
    if ha > 1e-9:
        return points * np.sqrt(1.0 / ha)
    return points

def construct():
    t0 = time.time()
    deadline = t0 + MAX_TIME
    
    # Baseline: Regular 14-gon
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    best_ratio = best_score # This is the ratio we want to maximize
    
    # We will optimize the Fourier coefficients.
    # Let's use up to K=3 harmonics (6 coefficients).
    # Initial guess: 0 (regular polygon)
    
    # Multi-start strategy
    # We try a few random perturbations to escape local minima.
    
    K = 3
    n_coeffs = 2 * K
    
    # Start 1: Zero coefficients (Regular polygon)
    start_coeffs = np.zeros(n_coeffs)
    res_coeffs, res_val = _optimize_shape(start_coeffs, deadline, maxiter=500)
    
    if res_coeffs is not None:
        pts = _fourier_to_points(res_coeffs)
        if pts is not None:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    # If we have time, try some random starts
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            
            # Small random perturbations
            # Scale: 0.05 is small enough to likely keep convexity
            start_coeffs = rng.uniform(-0.05, 0.05, n_coeffs)
            
            # Bias towards certain harmonics? 
            # For n=14, symmetry breaking might be useful.
            # Let's just use random uniform.
            
            res_coeffs, res_val = _optimize_shape(start_coeffs, deadline, maxiter=300)
            
            if res_coeffs is not None:
                pts = _fourier_to_points(res_coeffs)
                if pts is not None:
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts

    # If we have even more time, try a finer search around the best found
    # or a different set of harmonics.
    # But given the constraints, the above is a solid structural change.
    
    # Final check: ensure we return a valid configuration
    if best_pts is None or _score(best_pts) < 1e-6:
        best_pts = _regular_polygon(N)
        
    return _normalize_unit_area(best_pts)

# EVOLVE-BLOCK-END
