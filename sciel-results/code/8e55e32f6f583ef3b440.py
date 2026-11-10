# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _is_strictly_convex(polygon):
    """Check if polygon is strictly convex and has no repeated points."""
    n = len(polygon)
    if n < 3:
        return False
    
    p0 = polygon
    p1 = np.roll(polygon, -1, axis=0)
    p2 = np.roll(polygon, -2, axis=0)
    
    v1 = p1 - p0
    v2 = p2 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    if np.any(np.abs(cross) < 1e-10):
        return False
    
    signs = np.sign(cross)
    if not np.all(signs == signs[0]):
        return False
        
    # Check total turning is +-2*pi
    angles = np.arctan2(v2[:, 1], v2[:, 0]) - np.arctan2(v1[:, 1], v1[:, 0])
    # Normalize angles to [-pi, pi]
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    total_turn = np.sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
        
    return True

def _score_polygon_vectorized(polygon):
    """
    Calculate the score for a polygon.
    Returns the minimum vertex score.
    Returns -inf if invalid.
    """
    normalized = _normalize(polygon)
    if normalized is None:
        return -np.inf
        
    if not _is_strictly_convex(normalized):
        return -np.inf
        
    n = len(normalized)
    
    # Vectorized distance calculation
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)[:, :19]
    
    # Check d_0 < 0.05
    min_dists = sorted_dists[:, 0]
    invalid_mask = min_dists < 0.05
    
    # Calculate gaps
    gaps = np.diff(sorted_dists, axis=1)
    k_indices = np.argmin(gaps, axis=1)
    
    # Determine the 4 distances to use for each vertex
    indices = np.zeros((n, 4), dtype=int)
    
    mask_0 = (k_indices == 0)
    indices[mask_0, 0] = 0
    indices[mask_0, 1] = 1
    indices[mask_0, 2] = 2
    indices[mask_0, 3] = 3
    
    mask_17 = (k_indices == 17)
    indices[mask_17, 0] = 15
    indices[mask_17, 1] = 16
    indices[mask_17, 2] = 17
    indices[mask_17, 3] = 18
    
    mask_else = ~(mask_0 | mask_17)
    k_else = k_indices[mask_else]
    indices[mask_else, 0] = k_else - 1
    indices[mask_else, 1] = k_else
    indices[mask_else, 2] = k_else + 1
    indices[mask_else, 3] = k_else + 2
    
    row_indices = np.arange(n)[:, None]
    selected_dists = sorted_dists[row_indices, indices]
    
    d_k = sorted_dists[np.arange(n), k_indices]
    d_k_plus_1 = sorted_dists[np.arange(n), k_indices + 1]
    D = (d_k + d_k_plus_1) / 2.0
    
    invalid_mask |= (D < 0.1)
    
    ratios = selected_dists / D[:, None]
    raw_components = np.where(ratios >= 1.0, ratios, 1.0 / ratios)
    raw = np.mean(raw_components, axis=1)
    
    scores = np.minimum(1.0, 1.0 / raw)
    scores[invalid_mask] = -np.inf
    
    return np.min(scores)

def _score_polygon_smooth(polygon, beta=50.0):
    """
    Smooth surrogate for the min-over-vertices score using soft-min.
    Returns a smooth approximation of the minimum vertex score.
    """
    normalized = _normalize(polygon)
    if normalized is None:
        return -1e10
        
    if not _is_strictly_convex(normalized):
        return -1e10
        
    n = len(normalized)
    
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)[:, :19]
    
    min_dists = sorted_dists[:, 0]
    invalid_mask = min_dists < 0.05
    
    gaps = np.diff(sorted_dists, axis=1)
    k_indices = np.argmin(gaps, axis=1)
    
    indices = np.zeros((n, 4), dtype=int)
    
    mask_0 = (k_indices == 0)
    indices[mask_0, 0] = 0
    indices[mask_0, 1] = 1
    indices[mask_0, 2] = 2
    indices[mask_0, 3] = 3
    
    mask_17 = (k_indices == 17)
    indices[mask_17, 0] = 15
    indices[mask_17, 1] = 16
    indices[mask_17, 2] = 17
    indices[mask_17, 3] = 18
    
    mask_else = ~(mask_0 | mask_17)
    k_else = k_indices[mask_else]
    indices[mask_else, 0] = k_else - 1
    indices[mask_else, 1] = k_else
    indices[mask_else, 2] = k_else + 1
    indices[mask_else, 3] = k_else + 2
    
    row_indices = np.arange(n)[:, None]
    selected_dists = sorted_dists[row_indices, indices]
    
    d_k = sorted_dists[np.arange(n), k_indices]
    d_k_plus_1 = sorted_dists[np.arange(n), k_indices + 1]
    D = (d_k + d_k_plus_1) / 2.0
    
    invalid_mask |= (D < 0.1)
    
    ratios = selected_dists / D[:, None]
    raw_components = np.where(ratios >= 1.0, ratios, 1.0 / ratios)
    raw = np.mean(raw_components, axis=1)
    
    scores = np.minimum(1.0, 1.0 / raw)
    scores[invalid_mask] = -1e10
    
    # Soft-min: -log(sum(exp(-beta * s))) / beta
    valid_scores = scores[scores > -1e9]
    if len(valid_scores) == 0:
        return -1e10
    shifted = valid_scores - np.max(valid_scores)
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted)))
    soft_min = np.max(valid_scores) - log_sum_exp / beta
    
    return soft_min

def _angle_to_polygon(angle_deviations, radii, n=20):
    """Convert angle deviations and radii to a polygon."""
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    final_angles = base_angles + angle_deviations
    final_angles = np.sort(final_angles)
    x = radii * np.cos(final_angles)
    y = radii * np.sin(final_angles)
    return np.column_stack([x, y])

def _objective(params, n=20):
    """Negative score for minimization. Params are [angle_devs (20), radii (20)]."""
    angle_devs = params[:n]
    radii = params[n:]
    polygon = _angle_to_polygon(angle_devs, radii, n)
    score = _score_polygon_vectorized(polygon)
    if np.isneginf(score) or np.isnan(score):
        return 1e10
    return -score

def _objective_smooth(params, n=20, beta=50.0):
    """Negative smooth score for L-BFGS-B. Params are [angle_devs (20), radii (20)]."""
    angle_devs = params[:n]
    radii = params[n:]
    # Ensure radii are positive
    radii = np.maximum(radii, 0.05)
    polygon = _angle_to_polygon(angle_devs, radii, n)
    score = _score_polygon_smooth(polygon, beta=beta)
    if np.isnan(score) or score < -1e9:
        return 1e10
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0  # 630s limit, keep 40s margin
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_radii = np.ones(n)
    init_angle_devs = np.zeros(n)
    init_params = np.concatenate([init_angle_devs, init_radii])
    
    best_params = init_params.copy()
    best_score = _score_polygon_vectorized(_angle_to_polygon(init_angle_devs, init_radii, n))
    
    if np.isneginf(best_score):
        return _angle_to_polygon(init_angle_devs, init_radii, n)
        
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-restart with Nelder-Mead to find good local optima
    noise_scales = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    for noise_scale in noise_scales:
        if time.time() > deadline - 120.0:
            break
            
        if noise_scale == 0.0:
            current_params = init_params.copy()
        else:
            current_params = init_params.copy()
            current_params[:n] += rng.normal(0, noise_scale, size=n)
            current_params[n:] += rng.normal(0, noise_scale * 0.5, size=n)
            current_params[n:] = np.maximum(current_params[n:], 0.1)
        
        current_score = _score_polygon_vectorized(_angle_to_polygon(current_params[:n], current_params[n:], n))
        if not np.isneginf(current_score) and current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            
        try:
            res = minimize(_objective, current_params, args=(n,), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < -best_score:
                res_score = _score_polygon_vectorized(_angle_to_polygon(res.x[:n], res.x[n:], n))
                if not np.isneginf(res_score) and res_score > best_score:
                    best_score = res_score
                    best_params = res.x.copy()
                    
        except Exception:
            pass
            
    # Phase 2: Smooth L-BFGS-B polish using soft-min surrogate
    if time.time() < deadline - 60.0:
        # Try multiple beta values for the soft-min
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 30.0:
                break
            
            current_params = best_params.copy()
            # Ensure radii are positive for L-BFGS-B
            current_params[n:] = np.maximum(current_params[n:], 0.05)
            
            try:
                res = minimize(_objective_smooth, current_params, args=(n, beta), method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
                
                res_score = _score_polygon_vectorized(_angle_to_polygon(res.x[:n], np.maximum(res.x[n:], 0.05), n))
                if not np.isneginf(res_score) and res_score > best_score:
                    best_score = res_score
                    best_params = res.x.copy()
                    
            except Exception:
                pass
    
    # Phase 3: Coordinate ascent polishing on the best found parameters
    if time.time() < deadline - 20.0:
        current_params = best_params.copy()
        current_score = best_score
        step_size = 0.02
        no_improve_count = 0
        
        for iteration in range(3000):
            if time.time() > deadline - 10.0:
                break
            if no_improve_count > 80:
                break
                
            improved = False
            for i in range(2 * n):
                if time.time() > deadline - 10.0:
                    break
                for delta in [step_size, -step_size]:
                    trial_params = current_params.copy()
                    trial_params[i] += delta
                    if i >= n:
                        trial_params[i] = max(trial_params[i], 0.05)
                    
                    trial_polygon = _angle_to_polygon(trial_params[:n], trial_params[n:], n)
                    trial_score = _score_polygon_vectorized(trial_polygon)
                    
                    if not np.isneginf(trial_score) and trial_score > current_score + 1e-9:
                        current_params = trial_params
                        current_score = trial_score
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params.copy()
                        improved = True
                        break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                step_size *= 0.93
                if step_size < 1e-8:
                    break
    
    # Phase 4: Final high-precision polish with very small steps
    if time.time() < deadline - 5.0:
        current_params = best_params.copy()
        current_score = best_score
        step_size = 1e-4
        
        for iteration in range(5000):
            if time.time() > deadline - 3.0:
                break
                
            improved = False
            for i in range(2 * n):
                if time.time() > deadline - 3.0:
                    break
                for delta in [step_size, -step_size]:
                    trial_params = current_params.copy()
                    trial_params[i] += delta
                    if i >= n:
                        trial_params[i] = max(trial_params[i], 0.05)
                    
                    trial_polygon = _angle_to_polygon(trial_params[:n], trial_params[n:], n)
                    trial_score = _score_polygon_vectorized(trial_polygon)
                    
                    if not np.isneginf(trial_score) and trial_score > current_score + 1e-10:
                        current_params = trial_params
                        current_score = trial_score
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params.copy()
                        improved = True
                        break
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-10:
                    break
    
    return _angle_to_polygon(best_params[:n], best_params[n:], n)

# EVOLVE-BLOCK-END
