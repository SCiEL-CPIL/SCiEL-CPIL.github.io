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
    if np.any(cross == 0):
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
    deadline = start_time + 550.0
    
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
        if np.any(r_abs < 0.1):
            return 1.0
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + np.sum(r_abs**2)
        return -score
    
    best_score = -1.0
    best_points = None
    
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    init_score, valid = compute_metric(init_points)
    if valid:
        best_score = init_score
        best_points = init_points
    
    rng = np.random.default_rng(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            init_r = np.ones(n)
            init_off = np.zeros(n)
        else:
            init_r = 1.0 + rng.normal(0, 0.1, n)
            init_off = rng.normal(0, 0.1, n)
        
        init_params = np.concatenate([init_r, init_off])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'adaptive': True, 'xatol': 1e-3, 'fatol': 1e-3})
            
            if time.time() < deadline - 30:
                final_r = np.abs(res.x[:n])
                final_off = res.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points
                    
                    if time.time() < deadline - 15:
                        polish_params = np.concatenate([final_r, final_off])
                        res2 = minimize(objective, polish_params, method='Nelder-Mead',
                                       options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-5, 'fatol': 1e-5})
                        
                        if time.time() < deadline - 5:
                            polish_r = np.abs(res2.x[:n])
                            polish_off = res2.x[n:]
                            polish_points = get_points(polish_r, polish_off)
                            polish_score, valid2 = compute_metric(polish_points)
                            
                            if valid2 and polish_score > best_score:
                                best_score = polish_score
                                best_points = polish_points
                                
        except Exception:
            pass
    
    if best_points is None:
        return init_points
    
    final_score, valid = compute_metric(best_points)
    if not valid:
        return init_points
        
    return best_points
