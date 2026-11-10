import numpy as np
import time

N = 50

def _score(centers):
    """Compute the radius score: min of pairwise distances/2 and edge distances."""
    n = len(centers)
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_edge = np.min(edge_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_edge, min_pair)

def _neg_score(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return 1e10
    if not (np.all(centers >= -0.5) and np.all(centers <= 0.5)):
        return 1e10
    return -_score(centers)

def _smooth_neg_score(centers_flat, beta=100.0):
    """Smooth surrogate using log-sum-exp for the min operation."""
    centers = centers_flat.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return 1e10
    if not (np.all(centers >= -0.5) and np.all(centers <= 0.5)):
        return 1e10
    
    # Edge distances
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Collect all constraint values
    all_vals = np.concatenate([edge_dists, dists[np.triu_indices_from(dists, k=1)]])
    all_vals = all_vals[all_vals < np.inf]
    
    if len(all_vals) == 0:
        return 0.0
    
    # Soft min via log-sum-exp
    m = np.min(all_vals)
    log_sum_exp = m + np.log(np.sum(np.exp(all_vals - m)))
    soft_min = -log_sum_exp / beta + m  # Approximate the min
    
    # Actually, for soft-min: -1/beta * log(sum(exp(-beta * x_i)))
    # Let's do it properly:
    neg_beta_vals = -beta * all_vals
    max_val = np.max(neg_beta_vals)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_vals - max_val)))
    soft_min = -log_sum / beta
    
    return -soft_min

def _smooth_gradient(centers_flat, beta=100.0):
    """Gradient of the smooth surrogate."""
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    
    # Edge distances
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Build the full constraint vector and its gradient contributions
    # We'll compute weights for each constraint based on soft-min
    all_vals = np.concatenate([edge_dists, dists[np.triu_indices_from(dists, k=1)]])
    all_vals = all_vals[all_vals < np.inf]
    
    if len(all_vals) == 0:
        return np.zeros_like(centers_flat)
    
    neg_beta_vals = -beta * all_vals
    max_val = np.max(neg_beta_vals)
    exp_vals = np.exp(neg_beta_vals - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # These are the soft-min weights
    
    grad = np.zeros_like(centers_flat)
    
    # Edge constraint gradients
    edge_idx = 0
    for i in range(n):
        d_left = 0.5 - centers[i, 0]
        d_right = centers[i, 0] + 0.5
        d_bottom = 0.5 - centers[i, 1]
        d_top = centers[i, 1] + 0.5
        min_d = min(d_left, d_right, d_bottom, d_top)
        w = weights[edge_idx]
        if d_left <= min_d + 1e-12:
            grad[2*i] += w * (-1.0)
        if d_right <= min_d + 1e-12:
            grad[2*i] += w * 1.0
        if d_bottom <= min_d + 1e-12:
            grad[2*i+1] += w * (-1.0)
        if d_top <= min_d + 1e-12:
            grad[2*i+1] += w * 1.0
        edge_idx += 1
    
    # Pairwise constraint gradients
    pair_idx = edge_idx
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            if d < np.inf and d > 1e-15:
                w = weights[pair_idx]
                vec = centers[i] - centers[j]
                d_vec = vec / (2.0 * d)
                grad[2*i:2*i+2] += w * d_vec
                grad[2*j:2*j+2] -= w * d_vec
            pair_idx += 1
    
    return -grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    # Better hexagonal lattice initialization
    # Generate a dense hex lattice and select the best N points
    s = 0.15
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    
    # Select N points closest to origin
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    init_positions = points[indices].copy()
    
    # Scale to fit in [-0.5, 0.5]^2
    max_abs = np.max(np.abs(init_positions))
    if max_abs > 0:
        scale = 0.48 / max_abs
        init_positions = init_positions * scale
    
    best_centers = init_positions.copy()
    best_score = _score(best_centers)
    
    # Phase 1: Use smooth surrogate to navigate the landscape
    # Phase 2: Switch to exact nonsmooth objective for final optimization
    # Phase 3: High-precision polish
    
    from scipy.optimize import minimize
    
    # Phase 1: Smooth optimization with multiple restarts
    num_restarts_smooth = 8
    for restart in range(num_restarts_smooth):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            centers = best_centers.copy()
        elif restart == 1:
            jitter = 0.02
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.495, 0.495)
        elif restart == 2:
            # Different hex lattice spacing
            s2 = 0.18
            pts2 = []
            for j in range(-15, 16):
                for i in range(-15, 16):
                    x = i * s2 + (j % 2) * s2 / 2.0
                    y = j * s2 * np.sqrt(3.0) / 2.0
                    pts2.append([x, y])
            pts2 = np.array(pts2)
            d2c = np.sum(pts2**2, axis=1)
            idx2 = np.argsort(d2c)[:N]
            init2 = pts2[idx2].copy()
            ma2 = np.max(np.abs(init2))
            if ma2 > 0:
                init2 = init2 * (0.48 / ma2)
            centers = init2
        else:
            jitter = 0.04 / (restart + 1)
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.495, 0.495)
        
        try:
            result = minimize(
                _smooth_neg_score,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.495, 0.495)] * (2*N),
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
                jac=_smooth_gradient
            )
            
            new_score = _score(result.x.reshape(-1, 2))
            if new_score > best_score:
                best_score = new_score
                best_centers = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Phase 2: Exact nonsmooth optimization
    num_restarts_exact = 10
    for restart in range(num_restarts_exact):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            centers = best_centers.copy()
        elif restart == 1:
            jitter = 0.01
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
        else:
            jitter = 0.03 / (restart + 1)
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
        
        try:
            result = minimize(
                _neg_score,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2*N),
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100},
                jac=_gradient
            )
            
            new_score = _score(result.x.reshape(-1, 2))
            if new_score > best_score:
                best_score = new_score
                best_centers = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                _neg_score,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.49999, 0.49999)] * (2*N),
                options={'maxiter': 15000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100},
                jac=_gradient
            )
            new_score = _score(result.x.reshape(-1, 2))
            if new_score > best_score:
                best_score = new_score
                best_centers = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Ensure strict validity
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
