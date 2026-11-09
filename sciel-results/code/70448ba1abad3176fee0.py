import numpy as np
import time
from scipy.optimize import minimize

def compute_metric(points):
    """
    Computes the metric described in the task contract.
    points: (N, 2) array
    Returns: (metric, is_valid)
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False
    
    normalized = centered / mean_dist
    
    # Check for repeated points
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(normalized[i] - normalized[j])
            if d < min_pair_dist:
                min_pair_dist = d
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Vectorized pairwise distances
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_score = np.inf
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            score = 0.0
        else:
            diffs = np.diff(d_sorted)
            min_diff_idx = np.argmin(diffs)
            k = min_diff_idx
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                score = 0.0
            else:
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
        if score < min_score:
            min_score = score
            
    return min_score, True


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(r, off):
        r = np.abs(r)
        theta = angles + off
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def objective(params):
        if time.time() > deadline:
            return 1.0
        
        r = params[:n]
        off = params[n:]
        
        r_abs = np.abs(r)
        if np.any(r_abs < 0.05):
            return 1.0 + np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + 0.1 * np.sum(r_abs**2)
        return -score
    
    def polish_from_params(params, max_iter=2000, xatol=1e-5, fatol=1e-5):
        """Polish a set of parameters using Nelder-Mead."""
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'adaptive': True, 'xatol': xatol, 'fatol': fatol})
            return res.x
        except Exception:
            return params
    
    def score_params(params):
        r = np.abs(params[:n])
        off = params[n:]
        points = get_points(r, off)
        score, valid = compute_metric(points)
        if not valid:
            return -1.0
        return score
    
    best_score = -1.0
    best_params = None
    
    # Start with regular polygon (all r=1, all off=0)
    init_params = np.concatenate([np.ones(n), np.zeros(n)])
    init_score = score_params(init_params)
    if init_score > best_score:
        best_score = init_score
        best_params = init_params.copy()
    
    # Phase 1: Multiple Nelder-Mead restarts from diverse initializations
    rng = np.random.default_rng(12345)
    
    # Different noise scales for diversity
    noise_scales = [0.05, 0.1, 0.15, 0.2, 0.3]
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 120:
            break
        
        # Choose a noise scale
        ns = noise_scales[restart % len(noise_scales)]
        
        # Generate diverse initial conditions
        init_r = 1.0 + rng.normal(0, ns, n)
        init_off = rng.normal(0, ns, n)
        init_params = np.concatenate([init_r, init_off])
        
        # Polish with Nelder-Mead
        polished = polish_from_params(init_params, max_iter=3000, xatol=1e-5, fatol=1e-5)
        s = score_params(polished)
        
        if s > best_score:
            best_score = s
            best_params = polished.copy()
    
    # Phase 2: Fine-tune the best solution with tighter tolerances
    if best_params is not None and time.time() < deadline - 60:
        polished2 = polish_from_params(best_params, max_iter=5000, xatol=1e-7, fatol=1e-7)
        s2 = score_params(polished2)
        if s2 > best_score:
            best_score = s2
            best_params = polished2.copy()
    
    # Phase 3: Try a few more restarts with even tighter polish if time permits
    if time.time() < deadline - 40:
        for extra in range(3):
            if time.time() > deadline - 20:
                break
            ns = 0.05 + 0.05 * extra
            init_r = 1.0 + rng.normal(0, ns, n)
            init_off = rng.normal(0, ns, n)
            init_params = np.concatenate([init_r, init_off])
            polished = polish_from_params(init_params, max_iter=2000, xatol=1e-6, fatol=1e-6)
            s = score_params(polished)
            if s > best_score:
                best_score = s
                best_params = polished.copy()
    
    if best_params is None:
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    final_r = np.abs(best_params[:n])
    final_off = best_params[n:]
    final_points = get_points(final_r, final_off)
    
    final_score, valid = compute_metric(final_points)
    if not valid:
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    return final_points
