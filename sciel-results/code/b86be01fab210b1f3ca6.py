import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _make_ring_points(n_out, n_in, r_in, delta):
    """
    Create points for a two-ring configuration.
    Outer ring: n_out points, radius 1.0, angles 0, 2pi/n_out, ...
    Inner ring: n_in points, radius r_in, angles delta, delta + 2pi/n_in, ...
    """
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(delta, 2 * np.pi + delta, n_in, endpoint=False)
    
    pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    
    return np.vstack([pts_out, pts_in])

def _objective_ring(params, n_out, n_in):
    """
    Objective for ring parameter optimization.
    params: [r_in, delta]
    """
    r_in, delta = params
    if r_in < 0 or r_in > 1.0:
        return 1e10
    if delta < 0 or delta >= 2 * np.pi:
        return 1e10
        
    points = _make_ring_points(n_out, n_in, r_in, delta)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _optimize_ring(n_out, n_in, deadline):
    """
    Optimize the two-ring configuration for given n_out, n_in.
    Uses a grid search followed by local refinement.
    """
    t_start = time.time()
    time_remaining = deadline - t_start
    if time_remaining < 2:
        return None, -1e10

    # Grid search for initial best
    r_in_grid = np.linspace(0.1, 0.9, 100)
    delta_grid = np.linspace(0, 2 * np.pi / n_out, 50)
    
    best_score = -1e10
    best_params = None
    
    for r in r_in_grid:
        if time.time() > deadline - 1:
            break
        for d in delta_grid:
            points = _make_ring_points(n_out, n_in, r, d)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_params = [r, d]

    if best_params is None:
        return None, -1e10

    # Local refinement using Nelder-Mead on the 2D parameter space
    try:
        res = minimize(
            _objective_ring,
            best_params,
            args=(n_out, n_in),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res.fun < -1e9: # res.fun is -score, so if score > 1e9 it's bad, but we check if it's valid
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
    except Exception:
        pass

    points = _make_ring_points(n_out, n_in, best_params[0], best_params[1])
    return points, best_score

def _full_objective(x_flat):
    """
    Objective for full 26D optimization.
    """
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _refine_full(points, deadline):
    """
    Refine a given point configuration using full 26D local search.
    """
    t_start = time.time()
    if time.time() > deadline - 5:
        return points, _score(points)

    x0 = points.flatten()
    best_score = _score(points)
    best_x = x0.copy()

    # Nelder-Mead for robustness
    try:
        res = minimize(
            _full_objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        s = _score(res.x.reshape(N, 2))
        if s > best_score:
            best_score = s
            best_x = res.x.copy()
    except Exception:
        pass

    # L-BFGS-B for faster convergence if gradient is helpful (though non-smooth, it often works well in practice for this)
    if time.time() < deadline - 2:
        try:
            res_lbfgs = minimize(
                _full_objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            s = _score(res_lbfgs.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = res_lbfgs.x.copy()
        except Exception:
            pass

    return best_x.reshape(N, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_points = None
    best_score = 0.0

    def _update_best(points, score=None):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        if score is None:
            score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Candidate ring configurations (n_out, n_in)
    # The optimal for N=13 is often cited as 7-6 or 6-7 split.
    # We will test several splits.
    ring_configs = [
        (7, 6),
        (6, 7),
        (5, 8),
        (8, 5),
        (4, 9),
        (9, 4),
        (3, 10),
        (10, 3),
    ]

    # Step 1: Optimize ring configurations
    ring_results = []
    for n_out, n_in in ring_configs:
        if time.time() > deadline - 10:
            break
        points, score = _optimize_ring(n_out, n_in, deadline)
        if points is not None:
            ring_results.append((score, points))
            _update_best(points, score)
            print(f"Ring {n_out}-{n_in}: score={score:.6f}", flush=True)

    # Sort ring results by score descending
    ring_results.sort(key=lambda x: x[0], reverse=True)

    # Step 2: Refine the top ring configurations in the full 26D space
    # We take the top 3-5 to allow for different local optima
    top_k = 5
    for score, points in ring_results[:top_k]:
        if time.time() > deadline - 15:
            break
        refined_points, refined_score = _refine_full(points, deadline)
        _update_best(refined_points, refined_score)
        print(f"Refined from {score:.6f} to {refined_score:.6f}", flush=True)

    # Step 3: If no good ring config found, try a regular polygon as fallback
    if best_score < 0.01:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        _update_best(pts)
        refined_points, refined_score = _refine_full(pts, deadline)
        _update_best(refined_points, refined_score)

    # Step 4: Final high-precision polish on the absolute best
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                _full_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
