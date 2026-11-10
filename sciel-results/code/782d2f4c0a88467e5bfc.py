# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_scores(p):
    """
    Computes the minimum vertex score for a given polygon.
    """
    n = len(p)
    centroid = np.mean(p, axis=0)
    p_norm = p - centroid
    dists_c = np.linalg.norm(p_norm, axis=1)
    mean_dist = np.mean(dists_c)
    if mean_dist <= 0.01:
        return 0.0
    p_norm = p_norm / mean_dist
    
    # Convexity check
    e1 = np.roll(p_norm, -1, axis=0) - p_norm
    e2 = np.roll(p_norm, -2, axis=0) - np.roll(p_norm, -1, axis=0)
    crosses = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(np.abs(crosses) < 1e-9):
        return 0.0
    # Check sign consistency
    signs = np.sign(crosses)
    if np.any(signs[1:] != signs[0]):
        return 0.0
        
    # Repeated points check
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p_norm[i] - p_norm[j]) < 1e-6:
                return 0.0

    min_score = 1.0
    for i in range(n):
        vi = p_norm[i]
        d = np.linalg.norm(p_norm - vi, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            return 0.0
            
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
        if D < 0.1:
            return 0.0
            
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
            
        d_sel = d_sorted[idx]
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        if score < min_score:
            min_score = score
            if min_score == 0.0:
                return 0.0
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def objective_flat(p_flat):
        p = p_flat.reshape(n, 2)
        score = _compute_scores(p)
        return -score
        
    start_time = time.time()
    
    try:
        # Nelder-Mead is robust to non-smoothness
        res = minimize(objective_flat, p0.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 10000, 'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if time.time() - start_time > 500:
            # If it's taking too long, we might want to break, but minimize doesn't support it easily.
            # We rely on maxfev.
            pass

        best_p = res.x.reshape(n, 2)
        best_score = -res.fun
    except Exception:
        best_p = p0
        best_score = _compute_scores(p0)
        
    return best_p

# EVOLVE-BLOCK-END
