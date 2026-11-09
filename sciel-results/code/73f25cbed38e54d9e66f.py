import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _normalize(points):
    """Normalize points: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    dists = np.linalg.norm(pts, axis=1)
    mean_d = np.mean(dists)
    if mean_d <= 0.01:
        return None
    pts = pts / mean_d
    return pts

def _turn_signs(points):
    """Check that all turns have the same nonzero sign."""
    n = len(points)
    signs = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-15:
            return None  # zero turn
        signs.append(1 if cross > 0 else -1)
    if len(set(signs)) != 1:
        return None
    return signs[0]

def _is_valid(points):
    """Check if points form a strictly convex polygon."""
    pts = _normalize(points)
    if pts is None:
        return False
    # Check no repeated points
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return False
    # Check convexity
    sign = _turn_signs(pts)
    if sign is None:
        return False
    # Check total turning is ±2π (for a simple closed convex curve, this is automatic
    # if all turns have the same sign and the curve is closed, but let's verify)
    # Actually for a convex polygon, the total turning is always 2π (or -2π).
    # The sign check ensures consistency.
    return True

def _vertex_scores(points):
    """Compute vertex scores for normalized points."""
    n = len(points)
    scores = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        sorted_dists = np.sort(dists)
        if sorted_dists[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = sorted_dists[1:] - sorted_dists[:-1]
        k = np.argmin(gaps)
        D = (sorted_dists[k] + sorted_dists[k + 1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            d_sel = sorted_dists[0:4]
        elif k == 17:
            d_sel = sorted_dists[15:19]
        else:
            d_sel = sorted_dists[k - 1:k + 3]
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def _get_points_from_weights(weights):
    """Get points on unit circle from arc weights."""
    w_sum = np.sum(weights)
    if w_sum <= 0:
        w_sum = 1e-6
    angles = np.cumsum(weights / w_sum * 2 * np.pi)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def _params_to_weights(params):
    """Convert 20 params to 20 positive weights summing to 2π."""
    w = params ** 2 + 0.01
    w = w / np.sum(w)
    return w * 2 * np.pi

def _soft_min(scores, beta):
    """Soft-min of scores using log-sum-exp for stability."""
    neg_scores = -scores
    a = beta * neg_scores
    a_max = np.max(a)
    sum_exp = np.sum(np.exp(a - a_max))
    log_sum = a_max + np.log(sum_exp)
    return log_sum / beta

def _objective(params, beta=50.0):
    """Objective: soft-min of vertex scores (to be minimized)."""
    weights = _params_to_weights(params)
    points = _get_points_from_weights(weights)
    norm_pts = _normalize(points)
    if norm_pts is None:
        return 10.0
    scores = _vertex_scores(norm_pts)
    return _soft_min(scores, beta)

def _best_valid_from(params, deadline):
    """Evaluate exact score for params."""
    weights = _params_to_weights(params)
    points = _get_points_from_weights(weights)
    norm_pts = _normalize(points)
    if norm_pts is None:
        return 0.0, points
    if not _is_valid(norm_pts):
        return 0.0, points
    scores = _vertex_scores(norm_pts)
    min_score = np.min(scores)
    return min_score, points

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    n = 20
    x0 = np.ones(20)
    
    best_params = x0.copy()
    best_score = 0.0
    best_points = _get_points_from_weights(_params_to_weights(x0))
    
    # Evaluate initial
    init_score, _ = _best_valid_from(x0, deadline)
    best_score = init_score
    
    # Phase 1: Differential Evolution for global search
    bounds = [(0.1, 10.0) for _ in range(20)]
    
    if time.time() < deadline - 120:
        try:
            res_de = differential_evolution(
                _objective,
                bounds,
                args=(20.0,),  # beta=20 for smoother landscape
                maxiter=300,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.9,
                seed=42,
                polish=False
            )
            if time.time() < deadline - 60:
                score_de, pts_de = _best_valid_from(res_de.x, deadline)
                if score_de > best_score:
                    best_score = score_de
                    best_params = res_de.x.copy()
                    best_points = pts_de
        except Exception:
            pass
    
    # Phase 2: Local refinement with soft-min annealing
    rng = np.random.default_rng(123)
    
    # Generate restarts
    restarts = [best_params.copy()]
    
    # Random restarts
    for i in range(8):
        if time.time() > deadline - 30:
            break
        x_rand = rng.uniform(0.5, 3.0, size=20)
        restarts.append(x_rand)
    
    # Perturbations of best
    for i in range(8):
        if time.time() > deadline - 30:
            break
        x_pert = best_params + rng.normal(0, 0.15, size=20)
        x_pert = np.clip(x_pert, 0.1, 10.0)
        restarts.append(x_pert)
    
    for x_init in restarts:
        if time.time() > deadline - 15:
            break
        current_x = x_init.copy()
        for beta in [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            try:
                res_lm = minimize(
                    _objective,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current_x = res_lm.x
                score_lm, pts_lm = _best_valid_from(current_x, deadline)
                if score_lm > best_score:
                    best_score = score_lm
                    best_params = current_x.copy()
                    best_points = pts_lm
            except Exception:
                break
    
    # Phase 3: Final polish with high beta
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        try:
            res_final = minimize(
                _objective,
                best_params,
                args=(1000.0,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            score_final, pts_final = _best_valid_from(res_final.x, deadline)
            if score_final > best_score:
                best_score = score_final
                best_params = res_final.x.copy()
                best_points = pts_final
        except Exception:
            break
    
    # Ensure we return valid points
    final_score, final_pts = _best_valid_from(best_params, deadline)
    if final_score > 0:
        return final_pts
    return best_points
