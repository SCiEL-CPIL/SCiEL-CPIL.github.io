import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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
        if np.any(r_abs < 0.1):
            return 1.0
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + np.sum(r_abs**2)
        return -score
    
    def coordinate_ascent(points, max_iter=500, tol=1e-8):
        """Coordinate ascent on individual vertex positions to maximize min vertex score."""
        current_points = points.copy()
        current_score, valid = compute_metric(current_points)
        if not valid:
            return current_points, current_score
        
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                
                best_pos = current_points[i].copy()
                best_score_i = current_score
                
                centroid = np.mean(current_points, axis=0)
                radial_dir = current_points[i] - centroid
                radial_norm = np.linalg.norm(radial_dir)
                if radial_norm > 1e-10:
                    radial_dir = radial_dir / radial_norm
                else:
                    radial_dir = np.array([1.0, 0.0])
                
                # Try radial offsets with adaptive step sizes
                for ss in step_sizes:
                    for dr in [-ss, ss]:
                        new_pos = current_points[i] + dr * radial_dir
                        trial_points = current_points.copy()
                        trial_points[i] = new_pos
                        trial_score, trial_valid = compute_metric(trial_points)
                        if trial_valid and trial_score > best_score_i + tol:
                            best_score_i = trial_score
                            best_pos = new_pos
                            improved = True
                
                # Try angular offsets
                for ss in step_sizes:
                    for da in [-ss, ss]:
                        cos_a, sin_a = np.cos(da), np.sin(da)
                        rel = current_points[i] - centroid
                        new_rel = np.array([cos_a * rel[0] - sin_a * rel[1],
                                            sin_a * rel[0] + cos_a * rel[1]])
                        new_pos = centroid + new_rel
                        trial_points = current_points.copy()
                        trial_points[i] = new_pos
                        trial_score, trial_valid = compute_metric(trial_points)
                        if trial_valid and trial_score > best_score_i + tol:
                            best_score_i = trial_score
                            best_pos = new_pos
                            improved = True
                
                # Free 2D adjustment
                for ss in step_sizes[:3]:
                    for dx in [-ss, 0.0, ss]:
                        for dy in [-ss, 0.0, ss]:
                            if dx == 0.0 and dy == 0.0:
                                continue
                            new_pos = current_points[i] + np.array([dx, dy])
                            trial_points = current_points.copy()
                            trial_points[i] = new_pos
                            trial_score, trial_valid = compute_metric(trial_points)
                            if trial_valid and trial_score > best_score_i + tol:
                                best_score_i = trial_score
                                best_pos = new_pos
                                improved = True
                
                if improved:
                    current_points[i] = best_pos
                    current_score = best_score_i
            
            if not improved:
                break
        
        return current_points, current_score
    
    def gradient_refine(points, max_iter=200, tol=1e-9):
        """Use finite-difference gradients to refine via L-BFGS-B on smoothed objective."""
        current_points = points.copy()
        current_score, valid = compute_metric(current_points)
        if not valid:
            return current_points, current_score
        
        # Convert to parameter vector: angles + radii in polar coords around centroid
        def points_to_params(pts):
            centroid = np.mean(pts, axis=0)
            centered = pts - centroid
            r = np.linalg.norm(centered, axis=1)
            theta = np.arctan2(centered[:, 1], centered[:, 0])
            return np.concatenate([r, theta])
        
        def params_to_points(params):
            r = params[:n]
            theta = params[n:]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.column_stack([x, y])
        
        def smooth_objective(params):
            """Smoothed objective: use soft-min of vertex scores."""
            r = np.abs(params[:n])
            if np.any(r < 0.1):
                return 1.0 + np.sum((r - 0.1)**2)
            theta = params[n:]
            points = params_to_points(np.concatenate([r, theta]))
            score, valid = compute_metric(points)
            if not valid:
                return 1.0 + 0.01 * np.sum(r**2)
            return -score
        
        init_params = points_to_params(current_points)
        
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxfun': 10000})
            
            if time.time() > deadline - 15:
                return current_points, current_score
            
            final_r = np.abs(res.x[:n])
            final_theta = res.x[n:]
            final_points = params_to_points(np.concatenate([final_r, final_theta]))
            final_score, final_valid = compute_metric(final_points)
            
            if final_valid and final_score > current_score:
                return final_points, final_score
        except Exception:
            pass
        
        return current_points, current_score
    
    best_score = -1.0
    best_points = None
    
    # Start with regular polygon
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    init_score, valid = compute_metric(init_points)
    if valid:
        best_score = init_score
        best_points = init_points.copy()
    
    # Coordinate ascent from regular polygon
    if time.time() < deadline - 100:
        ca_points, ca_score = coordinate_ascent(init_points, max_iter=400)
        if ca_score > best_score:
            best_score = ca_score
            best_points = ca_points.copy()
        
        # Gradient refinement
        if time.time() < deadline - 80:
            gr_points, gr_score = gradient_refine(best_points, max_iter=300)
            if gr_score > best_score:
                best_score = gr_score
                best_points = gr_points.copy()
    
    # Differential evolution for global search
    if time.time() < deadline - 80:
        try:
            bounds = [(0.5, 2.0)] * n + [(-0.5, 0.5)] * n
            de_result = differential_evolution(
                objective, bounds, maxiter=80, tol=1e-5,
                popsize=20, seed=42, polish=False,
                init='sobol', mutation=(0.3, 1.0), recombination=0.7
            )
            
            if time.time() < deadline - 40:
                final_r = np.abs(de_result.x[:n])
                final_off = de_result.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
                    # Polish with coordinate ascent and gradient refinement
                    if time.time() < deadline - 25:
                        ca_points2, ca_score2 = coordinate_ascent(final_points, max_iter=250)
                        if ca_score2 > best_score:
                            best_score = ca_score2
                            best_points = ca_points2.copy()
                        
                        if time.time() < deadline - 15:
                            gr_pts2, gr_sc2 = gradient_refine(best_points, max_iter=200)
                            if gr_sc2 > best_score:
                                best_score = gr_sc2
                                best_points = gr_pts2.copy()
                            
        except Exception:
            pass
    
    # Nelder-Mead refinement from best found
    if best_points is not None and time.time() < deadline - 40:
        centroid = np.mean(best_points, axis=0)
        centered = best_points - centroid
        r_vals = np.linalg.norm(centered, axis=1)
        off_vals = np.arctan2(centered[:, 1], centered[:, 0]) - angles
        off_vals = np.unwrap(off_vals)
        init_params = np.concatenate([r_vals, off_vals])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 8000, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if time.time() < deadline - 15:
                final_r = np.abs(res.x[:n])
                final_off = res.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
                    # Polish with coordinate ascent and gradient refinement
                    if time.time() < deadline - 10:
                        ca_pts, ca_sc = coordinate_ascent(best_points, max_iter=200)
                        if ca_sc > best_score:
                            best_score = ca_sc
                            best_points = ca_pts.copy()
                        
                        if time.time() < deadline - 5:
                            gr_pts, gr_sc = gradient_refine(best_points, max_iter=200)
                            if gr_sc > best_score:
                                best_score = gr_sc
                                best_points = gr_pts.copy()
                            
        except Exception:
            pass
    
    # Additional Nelder-Mead restarts with diverse initial conditions
    rng = np.random.default_rng(123)
    restart_configs = [
        (0.1, 0.1),   # small perturbation
        (0.2, 0.2),   # medium perturbation
        (0.3, 0.15),  # larger radial, smaller angular
        (0.15, 0.3),  # smaller radial, larger angular
    ]
    
    for restart_idx, (sigma_r, sigma_a) in enumerate(restart_configs):
        if time.time() > deadline - 30:
            break
        
        # Use different seeds for diversity
        local_rng = np.random.default_rng(123 + restart_idx * 7)
        init_r = 1.0 + local_rng.normal(0, sigma_r, n)
        init_off = local_rng.normal(0, sigma_a, n)
        init_params = np.concatenate([init_r, init_off])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if time.time() < deadline - 10:
                final_r = np.abs(res.x[:n])
                final_off = res.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
                    if time.time() < deadline - 5:
                        ca_pts, ca_sc = coordinate_ascent(best_points, max_iter=150)
                        if ca_sc > best_score:
                            best_score = ca_sc
                            best_points = ca_pts.copy()
                        
                        if time.time() < deadline - 3:
                            gr_pts, gr_sc = gradient_refine(best_points, max_iter=150)
                            if gr_sc > best_score:
                                best_score = gr_sc
                                best_points = gr_pts.copy()
                            
        except Exception:
            pass
    
    if best_points is None:
        return init_points
    
    # Final polish: one more round of coordinate ascent + gradient refinement
    if time.time() < deadline - 5:
        ca_final, ca_sc_final = coordinate_ascent(best_points, max_iter=100)
        if ca_sc_final > best_score:
            best_score = ca_sc_final
            best_points = ca_final.copy()
    
    if time.time() < deadline - 3:
        gr_final, gr_sc_final = gradient_refine(best_points, max_iter=100)
        if gr_sc_final > best_score:
            best_score = gr_sc_final
            best_points = gr_final.copy()
    
    final_score, valid = compute_metric(best_points)
    if not valid:
        return init_points
        
    return best_points
