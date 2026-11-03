"""Smooth surrogate optimization for sphere packing: log-sum-exp soft-min with high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _soft_min_radius(centers, beta=80.0):
    """
    Compute a smooth approximation to the minimum radius using log-sum-exp.
    r_soft = -1/beta * log( sum_{i<j} exp(-beta*d_ij) + sum_i exp(-beta*w_i) )
    where d_ij = |c_i - c_j|/2 and w_i = min_k (0.5 - |c_ik|)
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    
    # Pair distances (half-distance since radius = d/2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    pair_dists = 0.5 * np.sqrt(sq_dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_per_sphere = np.min(wall_dists, axis=1)
    
    # Combine all constraints into one array
    # We want to minimize the soft-min of all pair distances and wall distances
    all_dists = np.concatenate([pair_dists[np.triu_indices(n, k=1)], min_wall_per_sphere])
    
    # Log-sum-exp for numerical stability
    beta_val = beta
    shifted = -beta_val * all_dists
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    
    r_soft = -log_sum_exp / beta_val
    return r_soft

def _soft_min_radius_grad(centers, beta=80.0):
    """
    Compute the gradient of the soft-min radius with respect to centers.
    d(r_soft)/d(c_i) = sum_j p_j * d(d_j)/d(c_i)
    where p_j = exp(-beta*d_j) / sum_k exp(-beta*d_k)
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    pair_dists = 0.5 * np.sqrt(sq_dists)
    
    # Wall distances per sphere
    wall_dists = 0.5 - np.abs(centers)
    min_wall_per_sphere = np.min(wall_dists, axis=1)
    
    # Build the list of all constraints and their gradients
    # For each pair (i,j) with i<j: d_ij = 0.5*|c_i - c_j|
    # d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / |c_i - c_j|
    # d(d_ij)/d(c_j) = 0.5 * (c_j - c_i) / |c_i - c_j|
    
    # For wall: d(w_i)/d(c_ik) = -sign(c_ik) for the minimizing coordinate
    
    # Compute weights
    all_dists = np.concatenate([pair_dists[np.triu_indices(n, k=1)], min_wall_per_sphere])
    shifted = -beta * all_dists
    max_shifted = np.max(shifted)
    weights = np.exp(shifted - max_shifted)
    weights /= np.sum(weights)
    
    # Pair gradients
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_weights = weights[:len(i_idx)]
    
    for idx in range(len(i_idx)):
        i = i_idx[idx]
        j = j_idx[idx]
        w = pair_weights[idx]
        if w < 1e-10:
            continue
        d_ij = pair_dists[i, j]
        if d_ij < 1e-10:
            continue
        # d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / |c_i - c_j|
        # d(d_ij)/d(c_j) = 0.5 * (c_j - c_i) / |c_i - c_j|
        # But d_ij = 0.5 * |c_i - c_j|, so d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / |c_i - c_j|
        vec = centers[i] - centers[j]
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            continue
        # r_soft increases when d_ij increases
        # d(r_soft)/d(c_i) += w * d(d_ij)/d(c_i) = w * 0.5 * vec / norm
        # But wait: r_soft = -1/beta * log(sum exp(-beta*d)), so
        # d(r_soft)/d(d_j) = -p_j where p_j = w_j
        # Actually: r_soft = -1/beta * log_sum_exp(-beta * d)
        # d(r_soft)/d(d_j) = -1/beta * (-beta * exp(-beta*d_j)) / sum(exp(-beta*d_k))
        # = exp(-beta*d_j) / sum(exp(-beta*d_k)) = p_j
        # So d(r_soft)/d(c_i) = sum_j p_j * d(d_j)/d(c_i)
        
        # For pair (i,j): d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / |c_i - c_j|
        # d(d_ij)/d(c_j) = 0.5 * (c_j - c_i) / |c_i - c_j|
        
        grad[i] += w * 0.5 * vec / norm
        grad[j] += w * 0.5 * (-vec) / norm
    
    # Wall gradients
    wall_weights = weights[len(i_idx):]
    for i in range(n):
        w = wall_weights[i]
        if w < 1e-10:
            continue
        # Find which coordinate gives the min wall distance
        min_coord = np.argmin(wall_dists[i])
        sign = np.sign(centers[i, min_coord])
        if sign == 0:
            sign = 1.0
        # d(w_i)/d(c_ik) = -sign(c_ik)
        # d(r_soft)/d(c_ik) += p_i * d(w_i)/d(c_ik) = p_i * (-sign)
        grad[i, min_coord] += w * (-sign)
    
    return _soft_min_radius(centers, beta), grad

def _neg_soft_radius_and_grad(x, beta=80.0):
    """Objective and gradient for L-BFGS-B: minimize -r_soft."""
    centers = x.reshape(N, 3)
    r, grad = _soft_min_radius_grad(centers, beta)
    return -r, -grad.flatten()

def _optimize_smooth(centers, beta=80.0, time_limit=20.0):
    """Optimize using L-BFGS-B with smooth soft-min objective."""
    start = time.time()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return 1e10, np.zeros_like(x)
        return _neg_soft_radius_and_grad(x, beta)
    
    x0 = centers.flatten()
    bounds = [(-0.4999999, 0.4999999)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50, 'maxfun': 5000}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _sgld_step(centers, beta, lr, noise_scale, rng):
    """Single Stochastic Gradient Langevin Dynamics step."""
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    
    # Compute gradient of -r_soft (we want to maximize r_soft, so minimize -r_soft)
    _, grad = _soft_min_radius_grad(centers, beta)
    
    # SGLD: x_{t+1} = x_t + lr * grad + noise_scale * N(0, I)
    # grad is d(r_soft)/d(c), so we go in +grad direction to increase r_soft
    noise = rng.normal(0, 1, size=centers.shape)
    centers_new = centers + lr * grad + noise_scale * noise
    
    # Clip to stay in bounds
    centers_new = np.clip(centers_new, -0.4999999, 0.4999999)
    return centers_new

def _sgld_optimize(centers, beta=50.0, time_limit=20.0, seed=42):
    """Optimize using Stochastic Gradient Langevin Dynamics to escape local minima."""
    start = time.time()
    rng = np.random.RandomState(seed)
    c = centers.copy()
    
    # Adaptive learning rate and noise schedule
    n_steps = 2000
    base_lr = 0.001
    base_noise = 0.005
    
    for step in range(n_steps):
        if time.time() - start > time_limit:
            break
        
        # Anneal noise and learning rate
        progress = step / n_steps
        lr = base_lr * (1.0 - 0.5 * progress)
        noise = base_noise * (1.0 - 0.8 * progress)
        
        c = _sgld_step(c, beta, lr, noise, rng)
    
    return c

def _high_precision_polish(centers, time_limit=15.0):
    """
    High-precision polish: use a very large beta to approximate the true min,
    with tighter tolerances.
    """
    start = time.time()
    
    # Use progressively larger beta values
    betas = [100.0, 200.0, 400.0, 800.0]
    
    c = centers.copy()
    for beta in betas:
        if time.time() - start > time_limit - 2.0:
            break
        remaining = time_limit - (time.time() - start)
        if remaining < 1.0:
            break
        c = _optimize_smooth(c, beta=beta, time_limit=remaining)
        c = np.clip(c, -0.4999999, 0.4999999)
    
    return c

def _grid_init(n):
    """Grid initialization."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Random initialization inside cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct optimal packing of N spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_centers = _grid_init(N)
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: SGLD to escape local minima
    if time.time() < deadline - 80.0:
        time_remaining = deadline - time.time() - 75.0
        if time_remaining > 10.0:
            c_sgld = _sgld_optimize(best_centers, beta=50.0, time_limit=time_remaining, seed=42)
            c_sgld = np.clip(c_sgld, -0.4999999, 0.4999999)
            
            r_new = _compute_radius(c_sgld)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = c_sgld.copy()
    
    # Phase 2: Smooth optimization with moderate beta
    if time.time() < deadline - 50.0:
        time_remaining = deadline - time.time() - 45.0
        if time_remaining > 5.0:
            # Try multiple restarts
            for restart in range(3):
                if time.time() > deadline - 45.0:
                    break
                
                if restart == 0:
                    init_c = best_centers.copy()
                elif restart == 1:
                    rng_init = np.random.RandomState(42)
                    init_c = best_centers + rng_init.normal(0, 0.005, size=(N, 3))
                    init_c = np.clip(init_c, -0.49999, 0.49999)
                else:
                    init_c = _random_init(N, seed=100 + restart)
                
                c_opt = _optimize_smooth(init_c, beta=50.0, time_limit=min(15.0, time_remaining))
                c_opt = np.clip(c_opt, -0.4999999, 0.4999999)
                
                r_new = _compute_radius(c_opt)
                if r_new > best_radius:
                    best_radius = r_new
                    best_centers = c_opt.copy()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 20.0:
        time_remaining = deadline - time.time() - 15.0
        if time_remaining > 5.0:
            c_polished = _high_precision_polish(best_centers, time_limit=time_remaining)
            c_polished = np.clip(c_polished, -0.4999999, 0.4999999)
            
            r_new = _compute_radius(c_polished)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = c_polished.copy()
    
    # Phase 4: Final tight polish if time remains
    if time.time() < deadline - 5.0:
        time_remaining = deadline - time.time() - 3.0
        if time_remaining > 2.0:
            try:
                c_final = _optimize_smooth(best_centers, beta=500.0, time_limit=time_remaining)
                c_final = np.clip(c_final, -0.4999999, 0.4999999)
                
                r_new = _compute_radius(c_final)
                if r_new > best_radius:
                    best_radius = r_new
                    best_centers = c_final.copy()
            except Exception:
                pass
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    return best_centers
