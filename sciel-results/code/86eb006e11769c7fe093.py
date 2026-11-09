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
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, 0.0)
    min_pair_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Vectorized pairwise distances (reuse dist_matrix)
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

def compute_metric_smooth(points, temperature=0.01):
    """
    Smooth surrogate for the metric using soft-min.
    Returns a smooth scalar objective (higher is better).
    """
    N = points.shape[0]
    if N != 20:
        return 0.0
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    
    normalized = centered / mean_dist
    
    # Check convexity softly
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    # Soft penalty for non-convexity
    cross_abs_min = np.min(np.abs(cross))
    if cross_abs_min < 1e-10:
        return -1.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return -1.0
    
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Compute all vertex scores first
    vertex_scores = np.zeros(N)
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            vertex_scores[i] = 0.0
        else:
            diffs = np.diff(d_sorted)
            k = np.argmin(diffs)
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                vertex_scores[i] = 0.0
            else:
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                vertex_scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    
    # Use soft-min with a floor to keep gradient information
    # Replace exact zeros with a small positive value for smoothness
    eps = 1e-6
    scores_clipped = np.maximum(vertex_scores, eps)
    
    # Soft-min: -temperature * log(sum(exp(-scores/temperature)))
    # This gives a smooth approximation to min(scores)
    log_sum_exp = temperature * np.log(np.sum(np.exp(-scores_clipped / temperature)))
    soft_min = -log_sum_exp
    
    # Also add a penalty for any exact zero scores to push away from them
    zero_penalty = 0.1 * np.sum(vertex_scores == 0)
    
    return soft_min - zero_penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 15.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(r, off):
        r = np.abs(r)
        theta = base_angles + off
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def points_to_params(points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        r = np.linalg.norm(centered, axis=1)
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        off = theta - base_angles
        # Unwrap angles to be continuous
        off = np.unwrap(off)
        # Normalize offsets to [-pi, pi]
        off = (off + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([r, off])
    
    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1.0
        
        r = params[:n]
        off = params[n:]
        
        r_abs = np.abs(r)
        if np.any(r_abs < 0.1):
            return 1.0 + np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + 0.1 * np.sum(r_abs**2)
        return -score

    def objective_smooth(params, temperature=0.01):
        if time.time() > deadline - safety_margin:
            return 1.0
        
        r = params[:n]
        off = params[n:]
        
        r_abs = np.abs(r)
        if np.any(r_abs < 0.1):
            return 1.0 + np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        smooth_score = compute_metric_smooth(points, temperature)
        if smooth_score < -0.5:
            return 1.0 + 0.1 * np.sum(r_abs**2)
        return -smooth_score
    
    def local_search(points, max_iter=200, tol=1e-7):
        """
        Perform local search by directly optimizing vertex positions.
        This avoids the parameterization issues of r, theta.
        """
        current_points = points.copy()
        current_score, valid = compute_metric(current_points)
        if not valid:
            return current_points, current_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - safety_margin:
                break
                
            improved = False
            for i in range(n):
                if time.time() > deadline - safety_margin:
                    break
                
                best_pos = current_points[i].copy()
                best_score = current_score
                
                # Try small perturbations in 2D
                for dx in np.linspace(-0.05, 0.05, 11):
                    for dy in np.linspace(-0.05, 0.05, 11):
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = current_points[i] + np.array([dx, dy])
                        trial_points = current_points.copy()
                        trial_points[i] = new_pos
                        trial_score, trial_valid = compute_metric(trial_points)
                        if trial_valid and trial_score > best_score + tol:
                            best_score = trial_score
                            best_pos = new_pos
                            improved = True
                
                if improved:
                    current_points[i] = best_pos
                    current_score = best_score
            
            if not improved:
                break
        
        return current_points, current_score
    
    best_score = -1.0
    best_points = None
    
    rng = np.random.default_rng(12345)
    
    # Generate a variety of initial candidates
    candidates = []
    
    # 1. Regular polygon
    r0 = np.ones(n)
    off0 = np.zeros(n)
    candidates.append(np.concatenate([r0, off0]))
    
    # 2. Perturbed regular polygons with various noise scales
    for noise_scale in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        r_noise = rng.normal(0, noise_scale, n)
        off_noise = rng.normal(0, noise_scale * 0.5, n)
        candidates.append(np.concatenate([1.0 + r_noise, off_noise]))
    
    # 3. Harmonic perturbations (mode shapes)
    for freq in range(1, 10):
        for amp in [0.01, 0.05, 0.1, 0.2]:
            phase = rng.uniform(0, 2 * np.pi)
            r_harm = 1.0 + amp * np.sin(freq * base_angles + phase)
            off_harm = amp * 0.5 * np.cos(freq * base_angles + phase)
            candidates.append(np.concatenate([r_harm, off_harm]))
    
    # 4. Random structured perturbations
    for _ in range(50):
        r_rand = 1.0 + 0.3 * rng.random(n)
        off_rand = 0.1 * rng.random(n)
        candidates.append(np.concatenate([r_rand, off_rand]))
    
    # 5. Alternating radius patterns
    for amp in [0.05, 0.1, 0.2]:
        r_alt = np.array([1.0 + amp if i % 2 == 0 else 1.0 - amp for i in range(n)])
        off_alt = np.zeros(n)
        candidates.append(np.concatenate([r_alt, off_alt]))
    
    # Evaluate all candidates and keep the best ones
    scored_candidates = []
    for params in candidates:
        if time.time() > deadline - 60:
            break
        r_abs = np.abs(params[:n])
        off = params[n:]
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if valid:
            scored_candidates.append((score, params, points))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top 20 candidates for local optimization
    top_k = min(20, len(scored_candidates))
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 30:
            break
        
        # First, try Nelder-Mead with smooth surrogate
        try:
            res = minimize(objective_smooth, params, method='Nelder-Mead',
                           options={'maxiter': 800, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7})
            
            r_final = np.abs(res.x[:n])
            off_final = res.x[n:]
            points_final = get_points(r_final, off_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_points = points_final.copy()
                
        except Exception:
            pass
        
        # Then, refine with exact metric using Nelder-Mead
        if time.time() > deadline - 30:
            break
        try:
            res2 = minimize(objective, params, method='Nelder-Mead',
                            options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
            
            r_final2 = np.abs(res2.x[:n])
            off_final2 = res2.x[n:]
            points_final2 = get_points(r_final2, off_final2)
            score_final2, valid_final2 = compute_metric(points_final2)
            
            if valid_final2 and score_final2 > best_score:
                best_score = score_final2
                best_points = points_final2.copy()
                
        except Exception:
            pass
        
        # Then, do local search directly on vertex positions
        if time.time() > deadline - 20:
            break
        
        # Use the best point from NM as starting point for local search
        start_pts = points_final2 if valid_final2 else (points_final if valid_final else points)
        ls_points, ls_score = local_search(start_pts, max_iter=100)
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
    
    # If we still have time, do more random restarts with local search
    if best_points is None or best_score < 1.46:
        for restart in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            r_init = 1.0 + 0.2 * rng.normal(0, 1, n)
            off_init = 0.1 * rng.normal(0, 1, n)
            params_init = np.concatenate([r_init, off_init])
            
            r_abs = np.abs(params_init[:n])
            points_init = get_points(r_abs, params_init[n:])
            
            # Quick local search
            ls_points, ls_score = local_search(points_init, max_iter=50)
            
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points.copy()
    
    if best_points is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Final high-precision polish pass
    if time.time() < deadline - 10:
        # Use L-BFGS-B on vertex coordinates directly for fine-tuning
        def vertex_objective(x):
            pts = x.reshape(n, 2)
            s, v = compute_metric(pts)
            if not v:
                return 1.0
            return -s
        
        try:
            x0 = best_points.flatten()
            res_polish = minimize(vertex_objective, x0, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            pts_polish = res_polish.x.reshape(n, 2)
            score_polish, valid_polish = compute_metric(pts_polish)
            if valid_polish and score_polish > best_score:
                best_score = score_polish
                best_points = pts_polish.copy()
        except Exception:
            pass
        
        # Additional polish with very tight Nelder-Mead
        if time.time() < deadline - 5:
            try:
                params_polish = points_to_params(best_points)
                res_nm = minimize(objective, params_polish, method='Nelder-Mead',
                                  options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10})
                r_p = np.abs(res_nm.x[:n])
                off_p = res_nm.x[n:]
                pts_nm = get_points(r_p, off_p)
                score_nm, valid_nm = compute_metric(pts_nm)
                if valid_nm and score_nm > best_score:
                    best_score = score_nm
                    best_points = pts_nm.copy()
            except Exception:
                pass
    
    return best_points
