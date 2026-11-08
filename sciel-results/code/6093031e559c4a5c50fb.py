"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Calculate min triangle area from triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the normalized minimum triangle area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = _triangle_areas(points)
    return min_tri_area / hull_area

def _neg_score_vec(x):
    """Objective for minimization: negative score, x is flat vector of length 2N."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Initial guess: Regular N-gon
    angles = 2 * np.pi * np.arange(N) / N
    points_init = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points_init.copy()
    best_score = _score(best_points)
    
    # Stage 1: Differential Evolution (global search)
    # Bounds: x,y in [-2, 2] to allow some flexibility
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    def de_callback(xk, convergence):
        """Check time and update best."""
        nonlocal best_points, best_score
        if time.time() > deadline:
            raise ValueError("timeout")
        pts = xk.reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    try:
        de_result = differential_evolution(
            _neg_score_vec,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            init='latinhypercube',
            updating='immediate',
            callback=de_callback,
            polish=False  # We'll do local refinement separately
        )
    except ValueError as e:
        if "timeout" not in str(e):
            pass
    except Exception:
        pass
    
    # Check DE result
    if de_result is not None and hasattr(de_result, 'x'):
        candidate = de_result.x.reshape(N, 2)
        cs = _score(candidate)
        if cs > best_score:
            best_score = cs
            best_points = candidate.copy()
    
    # Stage 2: Local refinement with Nelder-Mead from best point found
    if time.time() < deadline:
        nm_options = {
            'maxiter': 3000,
            'xatol': 1e-10,
            'fatol': 1e-10,
            'adaptive': True
        }
        try:
            nm_result = minimize(
                _neg_score_vec,
                best_points.flatten(),
                method='Nelder-Mead',
                options=nm_options
            )
            candidate = nm_result.x.reshape(N, 2)
            cs = _score(candidate)
            if cs > best_score:
                best_score = cs
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Stage 3: If time remains, try a few more NM restarts from perturbed best
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(123)
        for attempt in range(8):
            if time.time() > deadline - 2.0:
                break
            noise = rng.normal(0, 0.05, (N, 2))
            start_pts = best_points + noise
            try:
                nm_result = minimize(
                    _neg_score_vec,
                    start_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                candidate = nm_result.x.reshape(N, 2)
                cs = _score(candidate)
                if cs > best_score:
                    best_score = cs
                    best_points = candidate.copy()
            except Exception:
                pass
    
    return best_points

# EVOLVE-BLOCK-END
