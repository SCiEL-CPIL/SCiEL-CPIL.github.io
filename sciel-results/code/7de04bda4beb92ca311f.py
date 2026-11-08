"""Rewritten program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

def _hull_area(p):
    """Return area of the convex hull of point set p."""
    try:
        hull = ConvexHull(p)
        return hull.volume
    except Exception:
        return 1e-10

def _min_triangle_area(p):
    """Return minimum triangle area (2*area) over all triples."""
    n = len(p)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Calculate twice the area of triangle ijk
                # Using determinant formula: 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
                # We drop the 0.5 factor as it cancels out in the ratio with hull area if we were consistent, 
                # but here we compare raw min area to raw hull area.
                # Actually, _hull_area returns the true area. 
                # So we must return the true minimum area.
                # Area = 0.5 * |det|
                a = 0.5 * abs((p[j][0] - p[i][0]) * (p[k][1] - p[i][1]) - 
                              (p[j][1] - p[i][1]) * (p[k][0] - p[i][0]))
                if a < min_a:
                    min_a = a
    return min_a

def _score(p):
    """Return min_triangle_area / convex_hull_area."""
    hull_area = _hull_area(p)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(p)
    return min_area / hull_area

def _polar_to_cartesian(polar_params):
    """
    Convert polar parameters (r1, t1, r2, t2, ...) to Cartesian coordinates.
    polar_params: array of shape (2*N,)
    We sort by angle to ensure a consistent ordering, which helps in avoiding
    self-intersecting polygons if we were to compute hulls, but here we just
    need the set of points. Sorting is not strictly necessary for area calculation
    but helps in reproducibility and potential future constraints.
    """
    n = len(polar_params) // 2
    radii = polar_params[0::2]
    thetas = polar_params[1::2]
    
    # Normalize angles to [0, 2*pi)
    thetas = np.mod(thetas, 2 * np.pi)
    
    # Sort by angle to maintain a "convex-like" ordering in parameter space
    # This doesn't change the set of points, but ensures the "first" point is at angle 0-ish
    sort_idx = np.argsort(thetas)
    radii = radii[sort_idx]
    thetas = thetas[sort_idx]
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack([x, y])

def _objective(x):
    """
    Objective function for Differential Evolution (minimization).
    We want to maximize the score, so we return -score.
    """
    pts = _polar_to_cartesian(x)
    s = _score(pts)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for the 150s limit
    
    # Bounds for differential evolution
    # Radii: [0.0, 1.0]
    # Angles: [0.0, 2*pi]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))      # Radius
        bounds.append((0.0, 2 * np.pi)) # Angle
    
    # Initial best guess: Regular polygon
    # Radii all 1.0, Angles equally spaced
    best_polar = np.zeros(2 * N)
    for i in range(N):
        best_polar[2*i] = 1.0
        best_polar[2*i+1] = i * 2 * np.pi / N
    
    best_pts = _polar_to_cartesian(best_polar)
    best_score = _score(best_pts)
    
    # Run Differential Evolution
    # maxiter is set high, but we rely on the time limit
    # popsize 15 is a good default for balance
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15,
            tol=1e-6,
            polish=True, # Use L-BFGS-B for final polish
            # We don't pass a callback for time checking directly, 
            # but DE is relatively fast. 
            # To be safe, we can run it in a way that we can interrupt or just trust the maxiter.
            # Given N=14, 28 variables. DE with popsize 15 does 15*28 evals per iter.
            # 200 iters = 3000 evals. Each eval is ~1000 triangle calcs. 3M ops. Fast.
        )
        
        if res.success or res.fun < -best_score:
            candidate_polar = res.x
            candidate_pts = _polar_to_cartesian(candidate_polar)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts
                best_polar = candidate_polar
                
    except Exception:
        pass

    # If DE didn't run or failed, try a few random restarts with local polish if time permits
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            # Random start
            random_polar = np.random.uniform(0, 1, size=N) # Radii
            random_thetas = np.random.uniform(0, 2*np.pi, size=N)
            rp = np.zeros(2*N)
            rp[0::2] = random_polar
            rp[1::2] = random_thetas
            
            # Quick local optimization using Powell (derivative-free, robust)
            # We need to wrap the objective to return positive value for minimization? 
            # No, minimize -score.
            from scipy.optimize import minimize
            def obj_local(x):
                return _objective(x)
            
            try:
                res_local = minimize(obj_local, rp, method='Powell', options={'maxiter': 50, 'xtol': 1e-4})
                if res_local.fun < -best_score:
                    candidate_pts = _polar_to_cartesian(res_local.x)
                    candidate_score = _score(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts
            except Exception:
                pass

    return best_pts
# EVOLVE-BLOCK-END
