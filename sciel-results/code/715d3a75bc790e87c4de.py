# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _softmin_surrogate(polygon, beta=50.0):
    """
    Smooth soft-min surrogate for the vertex score.
    Uses log-sum-exp to create a smooth approximation of min(scores).
    Returns a scalar value (higher is better).
    """
    normalized = _normalize(polygon)
    if normalized is None:
        return -np.inf
    
    if not _is_strictly_convex(normalized):
        return -np.inf
    
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
    scores[invalid_mask] = 0.0
    
    # Soft-min using log-sum-exp
    # softmin(s, beta) = -log(sum(exp(-s_i / beta))) / beta
    # As beta -> 0, this approaches min(s)
    # For numerical stability, subtract max first
    s = scores
    s_max = np.max(s)
    if s_max < 1e-15:
        return -np.inf
    
    # Use the fact that softmin = -logsumexp(-s/beta)/beta
    # logsumexp(-s/beta) = log(sum(exp(-s/beta)))
    # For stability: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    neg_s_over_beta = -s / beta
    neg_s_max = np.max(neg_s_over_beta)
    log_sum = neg_s_max + np.log(np.sum(np.exp(neg_s_over_beta - neg_s_max)))
    softmin_val = -log_sum / beta
    
    return softmin_val

def _angle_to_polygon(angle_deviations, radii, n=20):
    """Convert angle deviations and radii to a polygon."""
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    final_angles = base_angles + angle_deviations
    # Sort angles to maintain cyclic order
    final_angles = np.sort(final_angles)
    # Compute coordinates
    x = radii * np.cos(final_angles)
    y = radii * np.sin(final_angles)
    return np.column_stack([x, y])

def _objective(params, n=20, beta=50.0):
    """
    Negative soft-min surrogate for minimization.
    Params are [angle_devs (20), radii (20)].
    """
    angle_devs = params[:n]
    radii = params[n:]
    polygon = _angle_to_polygon(angle_devs, radii, n)
    score = _softmin_surrogate(polygon, beta=beta)
    if np.isneginf(score) or np.isnan(score):
        return 1e10
    return -score

def _objective_exact(params, n=20):
    """Negative exact score for minimization (for final polish)."""
    angle_devs = params[:n]
    radii = params[n:]
    polygon = _angle_to_polygon(angle_devs, radii, n)
    score = _score_polygon_vectorized(polygon)
    if np.isneginf(score) or np.isnan(score):
        return 1e10
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 580.0  # 600s limit, keep 20s margin
    
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
    
    # Phase 1: Multi-restart with soft-min surrogate
    # Use progressively smaller beta for finer optimization
    noise_scales = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for noise_scale in noise_scales:
        if time.time() > deadline - 120.0:
            break
            
        if noise_scale == 0.0:
            current_params = init_params.copy()
        else:
            current_params = init_params.copy()
            # Perturb angles
            current_params[:n] += rng.normal(0, noise_scale, size=n)
            # Perturb radii (keep positive)
            current_params[n:] += rng.normal(0, noise_scale * 0.5, size=n)
            current_params[n:] = np.maximum(current_params[n:], 0.1)
        
        # Try multiple betas for this restart
        for beta in betas:
            if time.time() > deadline - 100.0:
                break
                
            try:
                res = minimize(_objective, current_params, args=(n, beta), method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                
                if res.success or res.fun < -best_score:
                    res_score = _score_polygon_vectorized(_angle_to_polygon(res.x[:n], res.x[n:], n))
                    if not np.isneginf(res_score) and res_score > best_score:
                        best_score = res_score
                        best_params = res.x.copy()
                        # Use improved params as starting point for next beta
                        current_params = res.x.copy()
                        
            except Exception:
                pass
    
    # Phase 2: Coordinate ascent polishing on the best found parameters
    if time.time() < deadline - 15.0:
        current_params = best_params.copy()
        current_score = best_score
        step_size = 0.02
        no_improve_count = 0
        
        for iteration in range(3000):
            if time.time() > deadline - 10.0:
                break
            if no_improve_count > 30:
                break
                
            improved = False
            for i in range(2 * n):
                if time.time() > deadline - 8.0:
                    break
                for delta in [step_size, -step_size]:
                    trial_params = current_params.copy()
                    trial_params[i] += delta
                    # Ensure radii are positive
                    if i >= n:
                        trial_params[i] = max(trial_params[i], 0.1)
                    
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
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                step_size *= 0.92
                if step_size < 1e-8:
                    break
    
    # Phase 3: L-BFGS-B polish on vertex coordinates directly
    if time.time() < deadline - 20.0:
        # Convert current params to polygon
        current_polygon = _angle_to_polygon(best_params[:n], best_params[n:], n)
        
        def _vertex_objective(verts_flat):
            verts = verts_flat.reshape(-1, 2)
            score = _score_polygon_vectorized(verts)
            if np.isneginf(score) or np.isnan(score):
                return 1e10
            return -score
        
        try:
            res = minimize(_vertex_objective, current_polygon.flatten(),
                           method='L-BFGS-B',
                           options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-15})
            
            res_polygon = res.x.reshape(-1, 2)
            res_score = _score_polygon_vectorized(res_polygon)
            
            if not np.isneginf(res_score) and res_score > best_score:
                best_score = res_score
                # Convert back to angle/radius params for consistency
                # Find centroid and radii
                centroid = res_polygon.mean(axis=0)
                centered = res_polygon - centroid
                radii = np.linalg.norm(centered, axis=1)
                base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
                angles = np.arctan2(centered[:, 1], centered[:, 0])
                # Unwrap angles to maintain cyclic order
                angle_diffs = np.diff(angles)
                angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                unwrapped_angles = angles[0] + np.cumsum(np.concatenate([[0], angle_diffs]))
                # Make sure we have 20 angles
                if len(unwrapped_angles) == n:
                    angle_devs = unwrapped_angles - base_angles
                    best_params = np.concatenate([angle_devs, radii])
                    
        except Exception:
            pass
    
    # Final polygon
    final_polygon = _angle_to_polygon(best_params[:n], best_params[n:], n)
    
    # One last check - if we have a valid polygon, return it
    final_score = _score_polygon_vectorized(final_polygon)
    if np.isneginf(final_score):
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    return final_polygon

# EVOLVE-BLOCK-END
