"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                 (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective(params, n=N):
    """Objective function to minimize: negative score."""
    points = params.reshape(n, 2)
    return -_score(points)


def _regular_polygon(n, radius=1.0, center=(0, 0)):
    """Generate a regular n-gon centered at the given center."""
    angles = 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def _jittered_polygon(n, radius=1.0, jitter_amp=0.1, seed=42):
    """Generate a regular n-gon with radial jitter."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(n) / n
    radii = radius * (1.0 + jitter_amp * rng.randn(n))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _optimize_from_initial(points, time_limit=100.0):
    """Run Nelder-Mead optimization from the given initial points."""
    start_time = time.time()
    n = len(points)
    
    # Flatten points for optimization
    x0 = points.flatten()
    
    # Nelder-Mead options
    options = {
        'maxiter': 5000,
        'maxfev': 10000,
        'xatol': 1e-8,
        'fatol': 1e-10,
    }
    
    best_points = points.copy()
    best_score = _score(points)
    
    def callback(xk):
        nonlocal best_points, best_score, start_time
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        points_k = xk.reshape(n, 2)
        score_k = _score(points_k)
        if score_k > best_score:
            best_score = score_k
            best_points = points_k.copy()
        return False  # Continue
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', 
                         options=options, callback=callback)
        final_points = result.x.reshape(n, 2)
        final_score = _score(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    return best_points, best_score


def construct():
    """Construct 14 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin before 120s limit
    
    # Strategy: Try multiple initializations and optimize each
    # 1. Regular polygon
    # 2. Jittered polygons with different seeds and amplitudes
    # 3. Random points
    
    candidates = []
    
    # Regular 14-gon
    candidates.append(_regular_polygon(N, radius=1.0))
    
    # Jittered polygons
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        for amp in [0.05, 0.1, 0.15, 0.2]:
            candidates.append(_jittered_polygon(N, radius=1.0, jitter_amp=amp, seed=seed))
    
    # Random points in unit square
    rng = np.random.RandomState(42)
    for _ in range(5):
        candidates.append(rng.rand(N, 2))
    
    # Optimize each candidate
    best_points = None
    best_score = -1.0
    
    for i, init_points in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Estimate remaining time and allocate accordingly
        remaining = deadline - elapsed
        time_per_candidate = min(15.0, remaining / max(1, len(candidates) - i))
        
        opt_points, opt_score = _optimize_from_initial(init_points, time_limit=time_per_candidate)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # If we haven't found anything, fall back to regular polygon
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
        best_score = _score(best_points)
    
    # Final polish: one more optimization pass with more iterations if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        opt_points, opt_score = _optimize_from_initial(best_points, time_limit=deadline - elapsed - 5)
        if opt_score > best_score:
            best_points = opt_points.copy()
            best_score = opt_score
    
    return best_points

# EVOLVE-BLOCK-END
