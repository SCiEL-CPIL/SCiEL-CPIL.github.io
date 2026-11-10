import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the score A_min / A_hull for a set of points.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for finite points
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 0:
        return 0.0
        
    # Compute all triangle areas vectorized
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    x1, y1 = points[i, 0], points[i, 1]
    x2, y2 = points[j, 0], points[j, 1]
    x3, y3 = points[k, 0], points[k, 1]
    
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def _score_from_radii(radii: np.ndarray) -> float:
    """
    Compute score from radial perturbations of a regular 13-gon.
    """
    n = len(radii)
    angles = 2 * np.pi * np.arange(n) / n
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return _compute_score(points)

def _negative_score_from_radii(radii: np.ndarray) -> float:
    """
    Objective function for minimization (we maximize score, so minimize negative).
    """
    s = _score_from_radii(radii)
    if not np.isfinite(s):
        return 1e10
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points.
    """
    start_time = time.time()
    max_time = 600.0
    
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Initial guess: regular 13-gon
    initial_radii = np.ones(n)
    initial_points = np.column_stack((np.cos(2*np.pi*np.arange(n)/n), np.sin(2*np.pi*np.arange(n)/n)))
    best_points = initial_points.copy()
    best_score = _compute_score(best_points)
    
    # Define bounds for radii. 
    # To maintain convexity and reasonable aspect ratio, keep radii in [0.5, 1.5]
    bounds = [(0.5, 1.5)] * n
    
    # 1. Differential Evolution for global search
    # Use a subset of the time for DE
    de_time_limit = max_time * 0.6
    de_start = time.time()
    
    try:
        # maxiter is a loose limit; we rely on the time check inside the worker if possible,
        # but DE doesn't have a built-in time stop. We estimate iterations.
        # Each function call takes ~1-5ms. 
        # Let's set a reasonable number of iterations.
        # If we spend 360s on DE, and 1 eval is 2ms, we can do ~180k evals.
        # DE with popsize=10 (130 params total) needs ~1000-5000 iterations to converge.
        # Let's try a moderate population and iteration count.
        
        def time_guard(x):
            if time.time() - de_start > de_time_limit:
                raise TimeoutError()
            return _negative_score_from_radii(x)
            
        # We can't easily put a time guard in DE without wrapping the objective.
        # Let's just run DE with a fixed number of iterations that fits in the time budget.
        # Estimate: 2ms per eval. 360s / 0.002s = 180,000 evals.
        # DE with popsize=15 (195 individuals) * maxiter.
        # 180000 / 195 ~ 900 iterations.
        
        result_de = differential_evolution(
            _negative_score_from_radii, 
            bounds, 
            maxiter=1000, 
            popsize=15, 
            seed=42,
            tol=1e-6,
            init='sobol'
        )
        
        if result_de.success or result_de.fun < -best_score:
            radii_opt = result_de.x
            points_opt = np.column_stack((radii_opt * np.cos(2*np.pi*np.arange(n)/n), radii_opt * np.sin(2*np.pi*np.arange(n)/n)))
            score_opt = _compute_score(points_opt)
            if score_opt > best_score:
                best_points = points_opt
                best_score = score_opt
                initial_radii = radii_opt.copy()
                
    except Exception as e:
        pass # If DE fails, proceed with initial
    
    # 2. Local Refinement with L-BFGS-B
    # Use remaining time for local polish
    local_start = time.time()
    local_time_limit = max_time - (local_start - start_time) - 5 # 5s safety margin
    
    if local_time_limit > 10:
        # Try refining from the best point found so far
        # Also try a few random restarts if time permits
        candidates = [initial_radii]
        
        # Add a few random perturbations to the best solution
        rng = np.random.default_rng(seed=123)
        for _ in range(3):
            perturbed = initial_radii + rng.normal(0, 0.1, size=n)
            candidates.append(perturbed)
        
        # Also try the regular polygon again as a candidate (in case DE drifted)
        candidates.append(np.ones(n))
        
        for cand in candidates:
            if time.time() - local_start > local_time_limit - 5:
                break
            
            try:
                # L-BFGS-B is good for smooth problems, but our objective has a 'min' which is non-smooth.
                # However, in the neighborhood of the optimum, the min is usually determined by a specific set of triangles.
                # We can try it anyway.
                res = minimize(
                    _negative_score_from_radii, 
                    cand, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                if res.fun < -best_score:
                    radii_opt = res.x
                    points_opt = np.column_stack((radii_opt * np.cos(2*np.pi*np.arange(n)/n), radii_opt * np.sin(2*np.pi*np.arange(n)/n)))
                    score_opt = _compute_score(points_opt)
                    if score_opt > best_score:
                        best_points = points_opt
                        best_score = score_opt
            except Exception:
                pass
                
    # Final check: ensure the returned points are valid
    if best_points is None:
        return initial_points
        
    # Verify convexity and score one last time
    final_score = _compute_score(best_points)
    if final_score <= 0:
        # Fallback to regular polygon
        angles = 2 * np.pi * np.arange(n) / n
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points

if __name__ == "__main__":
    pts = heilbronn_convex13()
    print(f"Score: {_compute_score(pts)}")
    print(pts)
