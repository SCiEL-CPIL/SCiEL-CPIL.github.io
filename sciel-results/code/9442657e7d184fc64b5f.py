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
        k = np.argmin(diffs)
        
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

def _param_to_polygon(params):
    """Convert Fourier coefficients to polygon vertices.
    params: [r0, a1, b1, a2, b2, ..., a5, b5] (11 params)
    r0: mean radius, (a_k, b_k): Fourier coefficients for k=1..5
    """
    n = 20
    r0 = params[0]
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n) * r0
    for k in range(1, 6):
        a_k = params[1 + 2*(k-1)]
        b_k = params[2 + 2*(k-1)]
        radii += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _objective(params):
    polygon = _param_to_polygon(params)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    # Convexity penalty
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    if penalty > 1e-4:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score + penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 - 15.0  # Safety margin
    
    n = 20
    # Regular polygon as fallback
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Fourier parameterization: [r0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    # 11 parameters total
    bounds = [(0.8, 1.2)] + [(-0.1, 0.1)] * 10
    
    best_polygon = regular.copy()
    best_score = -1e9
    
    # Phase 1: Differential Evolution for global search
    try:
        if time.time() < deadline - 120:
            de_result = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=200,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < 100:  # Valid result
                best_params = de_result.x
                best_polygon = _param_to_polygon(best_params)
                pn = _normalize(best_polygon)
                if pn is not None and _is_strictly_convex(pn):
                    best_score = np.min(compute_scores(pn))
    except Exception:
        pass
    
    # Phase 2: Local polish with Nelder-Mead
    try:
        if time.time() < deadline - 30:
            # Start from best DE result or regular polygon params
            if best_score > 0:
                x0 = best_params
            else:
                x0 = np.array([1.0] + [0.0] * 10)
            
            polish_result = minimize(
                _objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if polish_result.fun < 100:
                pn = _normalize(_param_to_polygon(polish_result.x))
                if pn is not None and _is_strictly_convex(pn):
                    s = np.min(compute_scores(pn))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn
    except Exception:
        pass
    
    # Phase 3: Final high-precision polish if time allows
    try:
        if time.time() < deadline - 10 and best_score > 0.5:
            # Extract current params from best_polygon by fitting
            # For simplicity, use the last polished params
            final_result = minimize(
                _objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if final_result.fun < 100:
                pn = _normalize(_param_to_polygon(final_result.x))
                if pn is not None and _is_strictly_convex(pn):
                    s = np.min(compute_scores(pn))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn
    except Exception:
        pass
    
    # Return the best valid polygon
    if best_score > 0:
        pn = _normalize(best_polygon)
        if pn is not None and _is_strictly_convex(pn):
            return pn
    
    # Fallback to regular polygon
    return regular
