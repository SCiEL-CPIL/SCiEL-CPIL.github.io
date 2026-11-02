"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import itertools

def _compute_score(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0 or not np.isfinite(hull_area):
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    
    if not np.isfinite(min_area):
        return 0.0
    
    return min_area / hull_area


def _score_neg(pts):
    """Negative score for minimization."""
    return -_compute_score(pts)


def construct():
    deadline = time.time() + 140  # 140s safety margin
    
    best_pts = None
    best_score = -1.0
    
    # Strategy 1: Differential evolution for global search
    # Use symmetric initial guess: regular polygon
    def _regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Initial best: regular 13-gon
    reg_pts = _regular_polygon(N)
    reg_score = _compute_score(reg_pts)
    best_pts = reg_pts.copy()
    best_score = reg_score
    
    # Try differential evolution with multiple restarts
    bounds = [(-2, 2)] * (2 * N)
    
    def _de_objective(x):
        pts = x.reshape(N, 2)
        return _score_neg(pts)
    
    # Run DE with time limit
    de_kwargs = {
        'bounds': bounds,
        'maxiter': 100,
        'popsize': 10,
        'tol': 1e-4,
        'seed': 42,
        'init': 'latinhypercube'
    }
    
    try:
        result = differential_evolution(_de_objective, bounds, maxiter=50, popsize=8, 
                                         seed=42, tol=1e-3, init='latinhypercube',
                                         polish=False)
        if time.time() < deadline:
            de_pts = result.x.reshape(N, 2)
            de_score = _compute_score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
    except Exception:
        pass
    
    # Strategy 2: Local refinement using Nelder-Mead
    # Start from best points found so far
    if time.time() < deadline - 10:
        # Try refinement from regular polygon
        try:
            res = minimize(_score_neg, reg_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() < deadline:
                nm_pts = res.x.reshape(N, 2)
                nm_score = _compute_score(nm_pts)
                if nm_score > best_score:
                    best_score = nm_score
                    best_pts = nm_pts.copy()
        except Exception:
            pass
    
    # Try refinement from DE result
    if time.time() < deadline - 10:
        try:
            res = minimize(_score_neg, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-12})
            if time.time() < deadline:
                nm_pts = res.x.reshape(N, 2)
                nm_score = _compute_score(nm_pts)
                if nm_score > best_score:
                    best_score = nm_score
                    best_pts = nm_pts.copy()
        except Exception:
            pass
    
    # Strategy 3: Try multiple random restarts with local refinement
    if time.time() < deadline - 20:
        np.random.seed(123)
        for restart in range(5):
            if time.time() >= deadline - 15:
                break
            # Random perturbation around regular polygon
            noise = np.random.randn(N, 2) * 0.3
            init_pts = reg_pts + noise
            try:
                res = minimize(_score_neg, init_pts.flatten(), method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
                if time.time() >= deadline - 10:
                    break
                nm_pts = res.x.reshape(N, 2)
                nm_score = _compute_score(nm_pts)
                if nm_score > best_score:
                    best_score = nm_score
                    best_pts = nm_pts.copy()
            except Exception:
                pass
    
    # Strategy 4: Final high-precision polish
    if time.time() < deadline - 5 and best_pts is not None:
        try:
            res = minimize(_score_neg, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-14})
            if time.time() < deadline:
                final_pts = res.x.reshape(N, 2)
                final_score = _compute_score(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Normalize to unit area convex hull
    if best_pts is not None:
        try:
            hull = ConvexHull(best_pts)
            hull_area = hull.volume
            if hull_area > 0 and np.isfinite(hull_area):
                # Scale so that convex hull area is 1
                scale = 1.0 / np.sqrt(hull_area)
                best_pts = best_pts * scale
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
