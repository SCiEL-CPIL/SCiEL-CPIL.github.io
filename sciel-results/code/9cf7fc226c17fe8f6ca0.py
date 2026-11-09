import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time
import itertools

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate min triangle area
    # Vectorized calculation for speed
    # Points shape: (13, 2)
    # We need areas of all C(13,3) = 286 triangles
    
    # Generate indices
    indices = np.array(list(itertools.combinations(range(13), 3)))
    # indices shape: (286, 3)
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    # Cross product in 2D: (p2-p1) x (p3-p1)
    # Area = 0.5 * | (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def _objective(params):
    """
    Objective function for Differential Evolution.
    We maximize the score, so we minimize the negative score.
    To help DE navigate, we can use a soft penalty or just the raw negative score.
    Given the non-smooth nature, DE is robust.
    """
    points = params.reshape(13, 2)
    
    # Quick validity check
    if not np.all(np.isfinite(points)):
        return 1e6
        
    # Check for duplicates (degenerate triangles)
    # If any two points are too close, area is ~0
    # This is handled by the score function returning 0 or small number
    
    s = _score(points)
    
    if s <= 0:
        return 1e6
    
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Uses Differential Evolution on the full 26D coordinate space to maximize
    the ratio of minimum triangle area to convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 590 # Safety margin for 600s limit
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    init_points = np.column_stack((np.cos(theta0), np.sin(theta0)))
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # Define bounds for Differential Evolution
    # We can restrict the search space to a unit square or circle to avoid
    # scaling issues, as the metric is scale-invariant (ratio of areas).
    # However, DE works better with finite bounds.
    # Let's use [-1.5, 1.5] for x and y to allow some flexibility outside the unit circle
    # but keep the hull area reasonable.
    bounds = [(-1.5, 1.5)] * 26
    
    # Run Differential Evolution
    # DE is stochastic, so we use a fixed seed for reproducibility if needed, 
    # but we want to explore.
    # maxiter and popsize control the search effort.
    # tol controls convergence.
    
    try:
        # Use 'best1bin' strategy which is standard
        # The objective function is non-smooth, so DE is appropriate.
        # We will run DE for a certain number of iterations.
        # To maximize performance within the time limit, we can run DE
        # and monitor time.
        
        # DE options:
        # - popsize: 15 (default is 15 * len(bounds) = 390, which is large. 
        #   Let's use a smaller popsize like 50-100 to allow more generations)
        # - maxiter: We will let it run until time limit or convergence.
        # - tol: 1e-8
        
        # Since DE can be slow, we might run it for a fixed number of iterations
        # or use a callback to stop if time is running out.
        
        def callback(xk, convergence=True):
            # Check time
            if time.time() > deadline - 5:
                return True # Stop optimization
            return False

        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, # Limit iterations to prevent hanging if convergence is slow
            popsize=50, 
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.9,
            polish=False, # Polish uses L-BFGS-B which might get stuck in local minima of the non-smooth function
            callback=callback
        )
        
        if result.success or result.fun < 0:
            de_points = result.x.reshape(13, 2)
            if np.all(np.isfinite(de_points)):
                de_score = _score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                    
    except Exception as e:
        pass

    # If DE didn't improve much, or if we have time left, 
    # we can try a local refinement on the best points found using a 
    # more aggressive local search (like Nelder-Mead on the full 26D space, 
    # or a custom coordinate descent).
    # However, Nelder-Mead on 26D is expensive. 
    # Let's try a few random restarts of DE with different seeds if time permits.
    
    if time.time() < deadline - 30:
        # Run a second round of DE with a different seed and maybe tighter bounds
        # or different parameters to explore other basins.
        try:
            # Tighter bounds might help if the first DE found a good region
            # But for now, let's just try another seed.
            result2 = differential_evolution(
                _objective, 
                bounds, 
                seed=123, 
                maxiter=100, 
                popsize=50, 
                tol=1e-8,
                polish=False,
                callback=callback
            )
            if result2.fun < 0:
                de_points2 = result2.x.reshape(13, 2)
                if np.all(np.isfinite(de_points2)):
                    de_score2 = _score(de_points2)
                    if de_score2 > best_score:
                        best_score = de_score2
                        best_points = de_points2.copy()
        except Exception:
            pass

    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points

if __name__ == "__main__":
    pts = heilbronn_convex13()
    print(_score(pts))
