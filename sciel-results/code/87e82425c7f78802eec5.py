import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def compute_scores(polygon):
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        dists[i] = np.inf
        dists = np.sort(dists)
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def _normalize(polygon):
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
    if mean_dist <= 0.01:
        return None
    return polygon_norm / mean_dist

def _is_strictly_convex(polygon):
    v_curr = polygon
    v_next = np.roll(polygon, -1, axis=0)
    v_next2 = np.roll(polygon, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return np.all(cross > 1e-8)

def _params_to_polygon(params):
    n = 20
    angles = np.sort(params[:n])
    radii = params[n:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _convexity_penalty(polygon):
    v_curr = polygon
    v_next = np.roll(polygon, -1, axis=0)
    v_next2 = np.roll(polygon, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    neg = np.minimum(cross, 0)
    return np.sum(neg**2)

def _min_distance_penalty(polygon_norm):
    min_dist = np.inf
    for i in range(20):
        for j in range(i+1, 20):
            d = np.linalg.norm(polygon_norm[i] - polygon_norm[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-4:
        return 1.0 / max(min_dist, 1e-10)
    return 0.0

def _objective(params):
    polygon = _params_to_polygon(params)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    min_dist_pen = _min_distance_penalty(polygon_norm)
    if min_dist_pen > 0:
        return 500.0 + min_dist_pen
    
    penalty = _convexity_penalty(polygon_norm)
    if penalty > 1e-6:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score

def _soft_min_objective(params, temperature=0.05):
    """
    Smooth surrogate for the min-score objective using log-sum-exp.
    Returns a smooth approximation of -min(scores).
    """
    polygon = _params_to_polygon(params)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    min_dist_pen = _min_distance_penalty(polygon_norm)
    if min_dist_pen > 0:
        return 500.0 + min_dist_pen
    
    penalty = _convexity_penalty(polygon_norm)
    if penalty > 1e-6:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    
    # Soft-min: -min(s_i) ≈ -(1/T) * log(sum(exp(T * s_i)))  for large T
    # But we want to maximize the min, so minimize -min = soft-min of negatives
    # Actually, let's just use the log-sum-exp trick for the minimum:
    # min(s_i) ≈ (1/T) * log(sum(exp(T * s_i / T))) ... no, that's max.
    # For min: min(s_i) = -max(-s_i) ≈ -(1/T) * log(sum(exp(-T * s_i)))
    # So -min(s_i) ≈ (1/T) * log(sum(exp(-T * s_i)))
    
    # Handle zero scores
    scores_clipped = np.maximum(scores, 1e-10)
    neg_scores = -scores_clipped
    max_neg = np.max(neg_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min_val = (1.0 / temperature) * log_sum
    
    return soft_min_val

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = time.time() + 590.0  # Safety margin
    
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_polygon = regular.copy()
    best_score = -1e9
    best_params = np.concatenate([angles, np.ones(n)])
    
    # Phase 1: Differential Evolution for global search
    try:
        if time.time() < deadline - 120:
            bounds = [(0.0, 2*np.pi)] * n + [(0.3, 3.0)] * n
            
            de_result = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=200,
                tol=1e-5,
                popsize=20,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.fun < 100:
                pn = _params_to_polygon(de_result.x)
                pn_norm = _normalize(pn)
                if pn_norm is not None and _is_strictly_convex(pn_norm):
                    s = np.min(compute_scores(pn_norm))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn_norm
                        best_params = de_result.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local polish with Nelder-Mead on exact objective
    try:
        if time.time() < deadline - 60:
            x0_polish = best_params if best_score > 0 else np.concatenate([angles, np.ones(n)])
            
            polish_result = minimize(
                _objective, x0_polish,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if polish_result.fun < 100:
                pn = _params_to_polygon(polish_result.x)
                pn_norm = _normalize(pn)
                if pn_norm is not None and _is_strictly_convex(pn_norm):
                    s = np.min(compute_scores(pn_norm))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn_norm
                        best_params = polish_result.x.copy()
    except Exception:
        pass
    
    # Phase 3: Soft-min optimization for smoother landscape
    try:
        if time.time() < deadline - 30 and best_score > 0.5:
            # Use soft-min objective for a smoother optimization landscape
            x0_soft = best_params.copy()
            
            # Multiple temperatures, decreasing
            for temp in [0.1, 0.05, 0.02]:
                if time.time() >= deadline - 20:
                    break
                    
                soft_obj = lambda p: _soft_min_objective(p, temperature=temp)
                
                soft_result = minimize(
                    soft_obj, x0_soft,
                    method='Nelder-Mead',
                    options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                
                if soft_result.fun < 100:
                    pn = _params_to_polygon(soft_result.x)
                    pn_norm = _normalize(pn)
                    if pn_norm is not None and _is_strictly_convex(pn_norm):
                        s = np.min(compute_scores(pn_norm))
                        if s > best_score:
                            best_score = s
                            best_polygon = pn_norm
                            best_params = soft_result.x.copy()
                            x0_soft = soft_result.x.copy()
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish
    try:
        if time.time() < deadline - 5 and best_score > 0.5:
            final_result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if final_result.fun < 100:
                pn = _params_to_polygon(final_result.x)
                pn_norm = _normalize(pn)
                if pn_norm is not None and _is_strictly_convex(pn_norm):
                    s = np.min(compute_scores(pn_norm))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn_norm
    except Exception:
        pass
    
    # Return the best valid polygon
    if best_score > 0:
        pn = _normalize(best_polygon)
        if pn is not None and _is_strictly_convex(pn):
            return pn
    
    return regular
