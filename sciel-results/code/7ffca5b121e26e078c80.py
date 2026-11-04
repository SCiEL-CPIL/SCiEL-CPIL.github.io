"""Heilbronn convex regions: maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """Calculate convex hull area."""
    try:
        # Check if points are collinear or fewer than 3 points
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Exact score: min triangle area / hull area."""
    h = _hull_area(pts)
    if h <= 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    return float(min_area / h)

def _neg_score(x):
    """Negative score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_params(n_outer, n_inner, r_inner):
    """Initial parameterization: outer regular polygon + inner ring."""
    ao = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    ai = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([np.cos(ao), np.sin(ao)])
    inner = np.column_stack([r_inner * np.cos(ai), r_inner * np.sin(ai)])
    return np.vstack([outer, inner])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    best_score = -np.inf
    best_pts = _regular_polygon(N, 1.0).copy()
    
    def update_best(pts, score):
        nonlocal best_score, best_pts
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()

    # 1. Global Exploration with Differential Evolution
    # We constrain points to a box to prevent floating point issues, 
    # although the score is scale-invariant.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Use a smooth surrogate for DE to guide it, or just the exact score?
    # DE is robust to non-smooth functions, but it's slower. 
    # Given 150s budget, we can run DE on the exact negative score.
    # To speed up DE, we can use a population size that balances exploration and speed.
    
    de_results = []
    
    try:
        # Run DE on the exact objective
        res_de = differential_evolution(
            _neg_score,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-5,
            popsize=15,
            polish=False, # We will polish separately
            init='sobol',
            update_interval=1
        )
        if time.time() < deadline - 20:
            de_pts = res_de.x.reshape(N, 2)
            s = _score(de_pts)
            update_best(de_pts, s)
            de_results.append(de_pts)
    except Exception as e:
        pass

    # 2. Multi-start Local Refinement with Nelder-Mead
    # Nelder-Mead is derivative-free and handles non-smooth functions well.
    
    # Generate diverse initial points for local search
    initial_candidates = []
    
    # Regular polygon
    initial_candidates.append(_regular_polygon(N, 1.0))
    
    # Perturbed regular polygons
    rng = np.random.default_rng(123)
    for i in range(5):
        noise = rng.normal(0, 0.1, (N, 2))
        initial_candidates.append(_regular_polygon(N, 1.0) + noise)
        
    # Two-ring configurations
    for n_outer in [6, 7, 8]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.5, 0.7, 0.9]:
                initial_candidates.append(_two_ring_params(n_outer, n_inner, r_inner))
                
    # Add DE result if it exists
    if de_results:
        initial_candidates.append(de_results[0])

    # Score and sort initial candidates
    scored_candidates = []
    for pts in initial_candidates:
        if time.time() > deadline - 30:
            break
        s = _score(pts)
        scored_candidates.append((s, pts))
        update_best(pts, s)
        
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Refine top candidates using Nelder-Mead
    top_k = min(8, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline - 5:
            break
        s_init, pts_init = scored_candidates[i]
        
        try:
            res_nm = minimize(
                _neg_score,
                pts_init.flatten(),
                method='Nelder-Mead',
                options={
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'maxiter': 2000,
                    'adaptive': True
                }
            )
            res_pts = res_nm.x.reshape(N, 2)
            s_res = _score(res_pts)
            update_best(res_pts, s_res)
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
