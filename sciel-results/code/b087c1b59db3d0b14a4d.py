import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def normalize_polygon(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return points
    return centered / mean_dist

def get_points(deltas):
    """
    Generate 20 points on unit circle with cyclic order preserved.
    deltas: 19 values added to angles 1..19 (angle 0 is fixed at 0).
    """
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles.copy()
    if len(deltas) > 0:
        thetas[1:] += deltas
    
    # Wrap to [0, 2*pi) and sort to maintain cyclic order
    thetas = thetas % (2*np.pi)
    thetas = np.sort(thetas)
    
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def _vertex_score_exact(v_idx, points):
    """
    Calculate the exact score for a single vertex based on the task description.
    """
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
        
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = int(np.argmin(gaps))
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
        
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    if raw == 0:
        return 0.0
        
    return min(1.0, 1.0 / raw)

def _exact_metric(points):
    """Compute the exact minimum vertex score."""
    n = points.shape[0]
    scores = np.array([_vertex_score_exact(i, points) for i in range(n)])
    return np.min(scores)

def _soft_min(values, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(values)
    if m < 1e-10:
        return 0.0
    return m - np.log(np.sum(np.exp(-beta * (values - m)))) / beta

def _soft_metric(deltas, beta=50.0):
    """
    Smooth surrogate objective to minimize (negative soft-min of vertex scores).
    Uses a differentiable approximation of the min vertex score.
    """
    points = get_points(deltas)
    points_norm = normalize_polygon(points)
    n = points_norm.shape[0]
    
    # Compute all vertex scores (using exact per-vertex but combined with soft-min)
    scores = np.array([_vertex_score_exact(i, points_norm) for i in range(n)])
    
    # If any score is very small, the soft-min will be small
    soft = _soft_min(scores, beta)
    return -soft

def _exact_min_vertex_score(deltas):
    """Exact objective for final evaluation."""
    points = get_points(deltas)
    points_norm = normalize_polygon(points)
    return -_exact_metric(points_norm)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # Safety margin (60s buffer)
    
    n = 20
    x0 = np.zeros(19)
    bounds = [(-0.15, 0.15) for _ in range(19)]
    
    # Evaluate initial point with exact metric
    best_deltas = x0.copy()
    best_exact_score = -_exact_min_vertex_score(x0)
    
    # Phase 1: Global search with DE using soft metric
    if time.time() < deadline - 80:
        try:
            res_de = differential_evolution(
                _soft_metric,
                bounds,
                maxiter=120,
                popsize=12,
                tol=1e-6,
                polish=False,
                seed=42,
                updating='immediate',
                init='sobol',
            )
            if res_de.fun < best_exact_score:
                # Verify with exact metric
                exact_val = -_exact_min_vertex_score(res_de.x)
                if exact_val < best_exact_score:
                    best_exact_score = exact_val
                    best_deltas = res_de.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B using soft metric
    if time.time() < deadline - 40:
        try:
            res_local = minimize(
                _soft_metric,
                best_deltas,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            exact_val = -_exact_min_vertex_score(res_local.x)
            if exact_val < best_exact_score:
                best_exact_score = exact_val
                best_deltas = res_local.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try multiple random restarts for local optimization
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for i in range(10):
            if time.time() > deadline - 15:
                break
            perturbation = rng.normal(0, 0.03, size=19)
            start = np.clip(best_deltas + perturbation, -0.15, 0.15)
            try:
                res = minimize(
                    _soft_metric,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                exact_val = -_exact_min_vertex_score(res.x)
                if exact_val < best_exact_score:
                    best_exact_score = exact_val
                    best_deltas = res.x.copy()
            except Exception:
                pass
    
    # Phase 4: Final tight polish
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                _soft_metric,
                best_deltas,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            exact_val = -_exact_min_vertex_score(res_final.x)
            if exact_val < best_exact_score:
                best_exact_score = exact_val
                best_deltas = res_final.x.copy()
        except Exception:
            pass
    
    return get_points(best_deltas)
