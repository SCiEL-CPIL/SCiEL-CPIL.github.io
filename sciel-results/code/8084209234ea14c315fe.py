"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 115.0  # safety margin before 120s kill


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas using cross product
    # For points i, j, k: area = 0.5 * |cross(pj-pi, pk-pi)|
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = dx1 * dy2 - dy1 * dx2
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    if min_area == float('inf') or min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area


def _neg_score(params):
    """Negative score for minimization. params is 1D array of length 2*N."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def construct():
    t_start = time.time()
    
    # Initial guess: vertices of a regular 13-gon on unit circle, scaled
    # This gives a good starting point with spread-out points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    # Normalize to unit area
    try:
        hull0 = ConvexHull(x0)
        area0 = hull0.volume
        if area0 > 1e-12:
            scale = np.sqrt(1.0 / area0)
            x0 = x0 * scale
    except Exception:
        pass
    
    best_params = x0.flatten().copy()
    best_score = _score_points(x0)
    
    # Differential evolution search
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    def time_check():
        return (time.time() - t_start) > (DEADLINE - 10.0)
    
    try:
        result = differential_evolution(
            _neg_score,
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            mutation=(0.4, 1.0),
            recombination=0.7,
            polish=True,
            init='sobol',
            callback=None,
            updating='deferred',
            workers=1
        )
        
        if result.success or result.fun < -best_score:
            candidate = result.x.reshape(N, 2)
            score = _score_points(candidate)
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
    except Exception:
        pass
    
    # Local refinement with L-BFGS-B from best DE result
    if (time.time() - t_start) < (DEADLINE - 5.0):
        try:
            res2 = minimize(
                _neg_score,
                best_params,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 10000}
            )
            score2 = _score_points(res2.x.reshape(N, 2))
            if score2 > best_score:
                best_score = score2
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Additional local refinement from the regular polygon start
    if (time.time() - t_start) < (DEADLINE - 5.0):
        try:
            res3 = minimize(
                _neg_score,
                x0.flatten(),
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 10000}
            )
            score3 = _score_points(res3.x.reshape(N, 2))
            if score3 > best_score:
                best_score = score3
                best_params = res3.x.copy()
        except Exception:
            pass
    
    # Final: try a few random restarts if time permits
    rng = np.random.RandomState(42)
    attempts = 0
    while (time.time() - t_start) < (DEADLINE - 5.0) and attempts < 5:
        attempts += 1
        # Random perturbation around best
        perturbation = rng.normal(0, 0.1, size=(2 * N))
        start = best_params + perturbation
        try:
            res4 = minimize(
                _neg_score,
                start,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 200, 'maxfun': 5000}
            )
            score4 = _score_points(res4.x.reshape(N, 2))
            if score4 > best_score:
                best_score = score4
                best_params = res4.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(N, 2)
