"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 14
_NUM_TRIPLES = N * (N-1) * (N-2) // 6

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized. Returns array of shape (_NUM_TRIPLES,)."""
    n = len(points)
    idx = np.array(list(combinations(range(n), 3)))
    tri = points[idx]
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    try:
        areas = _all_triangle_areas(points)
        min_area = np.min(areas)
        hull_area = _convex_hull_area(points)
        if hull_area <= 0 or not np.isfinite(min_area):
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _params_to_points(params):
    """Convert (r_i, theta_i) params to 2D points.
    params: array of shape (2*N,) where params[2*i]=r_i, params[2*i+1]=theta_i
    Returns: array of shape (N, 2)
    """
    radii = params[0::2]
    angles = params[1::2]
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _points_to_params(points):
    """Convert 2D points to (r, theta) params."""
    x, y = points[:, 0], points[:, 1]
    radii = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    params = np.empty(2 * N)
    params[0::2] = radii
    params[1::2] = angles
    return params

def _neg_score_from_params(params):
    """Negative score for minimization. Returns large value if degenerate."""
    try:
        points = _params_to_points(params)
        # Check for near-duplicate points
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(points[i] - points[j]) < 1e-4:
                    return 1.0
        sc = _score(points)
        if sc <= 0:
            return 1.0
        return -sc
    except Exception:
        return 1.0

def _smooth_neg_score(params, temperature=0.01):
    """Smooth approximation of negative score for gradient-free optimization."""
    try:
        points = _params_to_points(params)
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(points[i] - points[j]) < 1e-4:
                    return 1.0
        areas = _all_triangle_areas(points)
        # Soft min: -log(sum(exp(-area/T))) / N * T
        min_area = -temperature * np.log(np.sum(np.exp(-areas / temperature))) / N
        hull_area = _convex_hull_area(points)
        if hull_area <= 0 or not np.isfinite(min_area):
            return 1.0
        return -min_area / hull_area
    except Exception:
        return 1.0

def _regular_polygon(n, rotation=0.0):
    """Generate a regular n-gon centered at origin."""
    angles = rotation + 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _two_rings(n_outer, n_inner, r_outer=1.0, r_inner=0.5, rot_outer=0.0, rot_inner=0.0):
    """Generate points on two concentric rings."""
    outer = r_outer * _regular_polygon(n_outer, rot_outer)
    inner = r_inner * _regular_polygon(n_inner, rot_inner)
    return np.vstack([outer, inner])

def _initial_candidates():
    """Generate diverse initial configurations."""
    candidates = []
    # Regular 14-gon
    candidates.append(_regular_polygon(N))
    # Two rings with various splits
    for no, ni, ri, rot in [(8,6,0.55,np.pi/6), (7,7,0.5,np.pi/7), (9,5,0.6,np.pi/9),
                             (10,4,0.65,np.pi/10), (6,8,0.5,np.pi/8), (11,3,0.7,np.pi/3)]:
        candidates.append(_two_rings(no, ni, r_outer=1.0, r_inner=ri, rot_inner=rot))
    # Three rings: 6 + 4 + 4
    ring1 = 1.0 * _regular_polygon(6, 0)
    ring2 = 0.6 * _regular_polygon(4, np.pi/4)
    ring3 = 0.3 * _regular_polygon(4, 0)
    candidates.append(np.vstack([ring1, ring2, ring3]))
    # Slightly perturbed regular polygon
    base = _regular_polygon(N)
    np.random.seed(42)
    perturbed = base + 0.05 * np.random.randn(N, 2)
    candidates.append(perturbed)
    return candidates

def _nelder_mead_refine(points, max_iter=2000):
    """Nelder-Mead refinement in Cartesian space."""
    x0 = points.flatten()
    result = minimize(_neg_score_from_points_flat, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    return result.x.reshape(N, 2), -result.fun

def _neg_score_from_points_flat(x_flat):
    """Score function accepting flat array for scipy minimize."""
    points = x_flat.reshape(N, 2)
    return _neg_score_from_params(_points_to_params(points))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Phase 1: Generate initial candidates and evaluate
    candidates = _initial_candidates()
    best_points = candidates[0].copy()
    best_score = _score(best_points)
    for cand in candidates[1:]:
        sc = _score(cand)
        if sc > best_score:
            best_score = sc
            best_points = cand.copy()

    # Phase 2: Differential evolution in param space (global search)
    if time.time() < deadline - 40:
        # Param bounds: radius in [0.1, 1.5], angle in [0, 2*pi]
        bounds = []
        for i in range(N):
            bounds.append((0.1, 1.5))   # radius
            bounds.append((0.0, 2.0*np.pi))  # angle

        # Run DE with limited iterations to stay within time budget
        elapsed = time.time() - start_time
        remaining = deadline - 40 - elapsed
        # Estimate: each DE eval takes ~5ms, so we can do ~remaining/0.005 evaluations
        # DE with popsize=20, N=28 params, maxiter ~ remaining/0.005/20
        max_iter_de = max(10, int(remaining / 0.05))
        max_iter_de = min(max_iter_de, 300)

        def _de_objective(params):
            if time.time() > deadline - 35:
                return 1.0
            return _neg_score_from_params(params)

        try:
            de_result = differential_evolution(
                _de_objective, bounds,
                seed=42, maxiter=max_iter_de, popsize=15,
                tol=1e-6, polish=False, init='sobol'
            )
            if de_result.fun < -best_score:
                de_points = _params_to_points(de_result.x)
                de_score = _score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead refinement from best point
    if time.time() < deadline - 15:
        try:
            refined_points, refined_score = _nelder_mead_refine(best_points, max_iter=3000)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
        except Exception:
            pass

    # Phase 4: Additional Nelder-Mead from other good candidates
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        if np.allclose(cand, best_points, atol=1e-5):
            continue
        sc = _score(cand)
        if sc > best_score * 0.9:
            try:
                rp, rs = _nelder_mead_refine(cand, max_iter=1500)
                if rs > best_score:
                    best_score = rs
                    best_points = rp.copy()
            except Exception:
                pass

    # Phase 5: Final fine-tuning
    if time.time() < deadline - 5:
        try:
            rp, rs = _nelder_mead_refine(best_points, max_iter=2000)
            if rs > best_score:
                best_score = rs
                best_points = rp.copy()
        except Exception:
            pass

    # Ensure valid output
    if _score(best_points) <= 0:
        best_points = _regular_polygon(N)

    return best_points
# EVOLVE-BLOCK-END
