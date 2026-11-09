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
    Smooth surrogate for the metric using soft-min over vertex scores.
    Returns a smooth scalar (higher is better) that approximates min vertex score.
    """
    N = points.shape[0]
    if N != 20:
        return -10.0
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return -10.0
    
    normalized = centered / mean_dist
    
    # Soft convexity check
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    cross_abs_min = np.min(np.abs(cross))
    if cross_abs_min < 1e-10:
        return -10.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return -10.0
    
    # Check for repeated points softly
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    if min_pair_dist < 1e-4:
        return -10.0
    
    # Compute per-vertex scores smoothly
    scores = np.zeros(N)
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            scores[i] = 0.0
        else:
            diffs = np.diff(d_sorted)
            min_diff_idx = np.argmin(diffs)
            k = min_diff_idx
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                scores[i] = 0.0
            else:
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    
    # Soft-min over vertex scores using log-sum-exp
    neg_scores = -scores / temperature
    max_neg = np.max(neg_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min = -log_sum * temperature
    
    return soft_min

def equidistant_points_polygon20() -> np.ndarray:
    """
    Direct approach: Start with a regular 20-gon and use symmetry-preserving
    parameterization to optimize. The key insight is that for a convex 20-gon
    with equidistant-like properties, we can parameterize by radius and angular
    offsets from the regular polygon, then use gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 15.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(params):
        """
        params: [r_0, ..., r_19, theta_0, ..., theta_19]
        where r_i are radii and theta_i are angular offsets
        """
        r = np.abs(params[:n])
        theta = base_angles + params[n:]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def points_to_params(points):
        """Convert points to parameterization (r, angular offsets)"""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        r = np.linalg.norm(centered, axis=1)
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        off = theta - base_angles
        off = np.unwrap(off)
        off = (off + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([r, off])
    
    def objective(params):
        """Objective to minimize: negative score + penalties"""
        if time.time() > deadline - safety_margin:
            return 1e10
        
        r = params[:n]
        r_abs = np.abs(r)
        
        # Penalty for small radii
        r_penalty = 0.0
        for ri in r_abs:
            if ri < 0.1:
                r_penalty += (0.1 - ri) ** 2
        
        points = get_points(params)
        score, valid = compute_metric(points)
        
        if not valid:
            return 100.0 + r_penalty + 0.01 * np.sum(r_abs ** 2)
        
        return -score + 0.001 * r_penalty
    
    def objective_smooth(params):
        """Smooth objective for gradient-based methods"""
        if time.time() > deadline - safety_margin:
            return 1e10
        
        r = params[:n]
        r_abs = np.abs(r)
        
        # Penalty for small radii
        r_penalty = 0.0
        for ri in r_abs:
            if ri < 0.1:
                r_penalty += (0.1 - ri) ** 2
        
        points = get_points(params)
        smooth_score = compute_metric_smooth(points, temperature=0.01)
        
        return -smooth_score + 0.001 * r_penalty
    
    # Initial candidates: regular polygon and perturbations
    best_score = -1.0
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular polygon
    r_reg = np.ones(n)
    off_reg = np.zeros(n)
    candidates.append(np.concatenate([r_reg, off_reg]))
    
    # 2. Small perturbations of regular polygon
    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        r_pert = 1.0 + scale * rng.normal(0, 1, n)
        off_pert = scale * 0.5 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_pert), off_pert]))
    
    # 3. Harmonic modes
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.05, 0.1, 0.15, 0.2]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                r_h = 1.0 + amp * np.sin(freq * base_angles + phase)
                off_h = amp * 0.3 * np.cos(freq * base_angles + phase)
                candidates.append(np.concatenate([np.abs(r_h), off_h]))
    
    # 4. Alternating patterns
    for amp in [0.05, 0.1, 0.15]:
        for period in [2, 4, 5]:
            r_alt = 1.0 + amp * np.sin(2 * np.pi * base_angles * period / (2 * np.pi))
            off_alt = amp * 0.5 * np.cos(2 * np.pi * base_angles * period / (2 * np.pi))
            candidates.append(np.concatenate([np.abs(r_alt), off_alt]))
    
    # 5. Random structured
    for _ in range(30):
        r_rand = 1.0 + 0.2 * rng.normal(0, 1, n)
        off_rand = 0.1 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_rand), off_rand]))
    
    # 6. Slightly non-uniform angular spacing
    for _ in range(20):
        r_uniform = np.ones(n)
        # Perturb angles slightly
        angle_pert = 0.05 * rng.normal(0, 1, n)
        angle_pert -= np.mean(angle_pert)  # Keep total angle = 2*pi
        off_custom = angle_pert
        candidates.append(np.concatenate([r_uniform, off_custom]))
    
    # Evaluate all candidates
    scored_candidates = []
    for i, params in enumerate(candidates):
        if time.time() > deadline - 120:
            break
        
        points = get_points(params)
        score, valid = compute_metric(points)
        if valid:
            scored_candidates.append((score, params, points))
    
    # Sort by score
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Optimize top candidates
    top_k = min(30, len(scored_candidates))
    
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 60:
            break
        
        # First: smooth gradient-based optimization to get close
        if time.time() > deadline - 30:
            break
        
        try:
            res_smooth = minimize(objective_smooth, params, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            r_s = np.abs(res_smooth.x[:n])
            off_s = res_smooth.x[n:]
            params_s = np.concatenate([r_s, off_s])
            points_s = get_points(params_s)
            score_s, valid_s = compute_metric(points_s)
            
            if valid_s and score_s > best_score:
                best_score = score_s
                best_params = params_s.copy()
        except Exception:
            pass
        
        # Then: Nelder-Mead on the exact objective
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
            
            r_final = np.abs(res.x[:n])
            off_final = res.x[n:]
            params_final = np.concatenate([r_final, off_final])
            points_final = get_points(params_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_params = params_final.copy()
                
        except Exception:
            pass
        
        # Local search on vertex positions for fine-tuning - more efficient version
        if time.time() > deadline - 10:
            break
        
        current_points = points.copy()
        current_score = score
        
        # Adaptive coordinate ascent with fewer iterations but smarter steps
        for iteration in range(30):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                
                best_pos = current_points[i].copy()
                best_local_score = current_score
                
                # Adaptive multi-scale: start coarse, refine
                for scale in [0.02, 0.01, 0.005, 0.002]:
                    # Use golden section search in 2D for better efficiency
                    for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                        dx = scale * np.cos(angle)
                        dy = scale * np.sin(angle)
                        new_pos = current_points[i] + np.array([dx, dy])
                        trial_points = current_points.copy()
                        trial_points[i] = new_pos
                        trial_score, trial_valid = compute_metric(trial_points)
                        if trial_valid and trial_score > best_local_score + 1e-9:
                            best_local_score = trial_score
                            best_pos = new_pos
                            improved = True
                
                if improved:
                    current_points[i] = best_pos
                    current_score = best_local_score
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_params = points_to_params(current_points)
    
    # Final high-precision polish
    if time.time() < deadline - 5 and best_params is not None:
        # Multiple polish passes with decreasing tolerance
        polish_params = best_params.copy()
        for tol in [1e-8, 1e-10, 1e-12]:
            if time.time() > deadline - 2:
                break
            try:
                res_final = minimize(objective, polish_params, method='Nelder-Mead',
                                      options={'maxiter': 300, 'adaptive': True, 'xatol': tol, 'fatol': tol})
                r_p = np.abs(res_final.x[:n])
                off_p = res_final.x[n:]
                polish_params = np.concatenate([r_p, off_p])
                points_p = get_points(polish_params)
                score_p, valid_p = compute_metric(points_p)
                
                if valid_p and score_p > best_score:
                    best_score = score_p
                    best_params = polish_params.copy()
            except Exception:
                pass
        
        # Final coordinate ascent polish
        if time.time() < deadline - 2:
            current_points = get_points(best_params)
            current_score = best_score
            
            for iteration in range(20):
                if time.time() > deadline - 1:
                    break
                
                improved = False
                for i in range(n):
                    if time.time() > deadline - 1:
                        break
                    
                    best_pos = current_points[i].copy()
                    best_local_score = current_score
                    
                    for scale in [0.005, 0.002, 0.001]:
                        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                            dx = scale * np.cos(angle)
                            dy = scale * np.sin(angle)
                            new_pos = current_points[i] + np.array([dx, dy])
                            trial_points = current_points.copy()
                            trial_points[i] = new_pos
                            trial_score, trial_valid = compute_metric(trial_points)
                            if trial_valid and trial_score > best_local_score + 1e-10:
                                best_local_score = trial_score
                                best_pos = new_pos
                                improved = True
                    
                    if improved:
                        current_points[i] = best_pos
                        current_score = best_local_score
                
                if not improved:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_params = points_to_params(current_points)
    
    # Return best points
    if best_params is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    return get_points(best_params)
