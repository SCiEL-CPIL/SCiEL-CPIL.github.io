"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

# Precompute all triples
_TRIPLES = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
_N_TRIPLES = len(_TRIPLES)

def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use all points as if they form a polygon (shouldn't happen for valid points)
        return 0.0

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    min_area = float('inf')
    for i, j, k in _TRIPLES:
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_area:
            min_area = area
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _negative_score(params):
    """Negative score for minimization."""
    points = params.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Initial seed: regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    seed_points = np.column_stack([np.cos(angles), np.sin(angles)])
    seed_score = _score(seed_points)
    best_points = seed_points.copy()
    best_score = seed_score
    
    # Also try a few other initial configurations
    for trial in range(5):
        if time.time() > deadline:
            break
        # Random points in unit circle
        rng = np.random.RandomState(42 + trial)
        r = np.sqrt(rng.random(N))
        theta = rng.uniform(0, 2*np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Differential evolution for global optimization
    # Parameters: 2*N coordinates
    bounds = [(-1, 1)] * (2 * N)
    
    try:
        result = differential_evolution(
            _negative_score,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=True,
            updating='immediate',
            callback=lambda x, convergence: (time.time() > deadline and (_ for _ in ()).throw(Exception("timeout")))
        )
        
        if result.fun is not None:
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
    except Exception:
        pass
    
    # Local refinement using Nelder-Mead on the best points found
    if time.time() < deadline - 5.0 and best_points is not None:
        from scipy.optimize import minimize
        try:
            result_lm = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result_lm.fun is not None:
                candidate_points = result_lm.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Final polish: try to improve with a tighter local search
    if time.time() < deadline - 2.0 and best_points is not None:
        from scipy.optimize import minimize
        try:
            result_lm2 = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'initial_simplex': None}
            )
            if result_lm2.fun is not None:
                candidate_points = result_lm2.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
