"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair, min_bound)

def _soft_min_obj_grad(x, n, beta):
    """
    Soft-min objective and gradient. x is flattened centers (2n values).
    Returns (obj_value, gradient_vector).
    
    The objective is -soft_min(min(pair_dists/2, bound_dists)) where
    soft_min uses log-sum-exp for numerical stability.
    """
    centers = x.reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Collect all constraint values: pair distances / 2 and boundary distances
    # For pairs: dists[i,j] / 2 for i < j
    # For bounds: min(0.5 - |x_i|, 0.5 - |y_i|)
    
    # Build arrays of values and their gradients
    # We'll compute soft_min = m - (1/beta) * log(sum(exp(-beta*(v_i - m))))
    # where v_i are all constraint values
    
    # Get pair values (upper triangle only to avoid double counting)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]  # shape (n*(n-1)/2,)
    pair_vals = pair_dists / 2.0
    
    # Boundary values
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    all_vals = np.concatenate([pair_vals, bound_vals])
    m = np.min(all_vals)
    
    # For numerical stability, shift by m
    shifted = all_vals - m
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    
    obj_val = -soft_min_val
    
    # Gradient computation
    # d(soft_min)/dx = -d(m)/dx + (1/beta) * d(log_sum)/dx
    # But since m is the min, and we're using the shifted form:
    # soft_min = m - (1/beta)*log(sum(exp(-beta*(v_i - m))))
    # Let S = sum(exp(-beta*(v_i - m)))
    # soft_min = m - (1/beta)*log(S)
    # d(soft_min)/dx = dm/dx - (1/beta) * (1/S) * dS/dx
    # dS/dx = sum_i exp(-beta*(v_i - m)) * (-beta) * (dv_i/dx - dm/dx)
    # This is complex. Let's use a simpler approach:
    
    # Actually, let's compute the gradient numerically-friendly way:
    # The soft min is a smooth approximation to the min.
    # weights_i = exp(-beta*(v_i - m)) / sum(exp(-beta*(v_j - m)))
    # d(soft_min)/dx = sum_i weights_i * dv_i/dx
    
    weights = exp_terms / np.sum(exp_terms)
    # weights shape: (n_pairs + n,)
    
    grad = np.zeros_like(x)
    
    # Pair contributions: for pair (i,j), v = dist(i,j)/2
    # d(dist(i,j))/d(centers[i]) = (centers[i] - centers[j]) / dist(i,j)
    # d(dist(i,j))/d(centers[j]) = (centers[j] - centers[i]) / dist(i,j)
    # So d(v)/d(centers[i]) = (centers[i] - centers[j]) / (2*dist(i,j))
    
    # pair_vals and weights for pairs
    pair_weights = weights[:len(pair_vals)]
    
    for k in range(len(pair_vals)):
        i, j = iu[0][k], iu[1][k]
        d = dists[i, j]
        if d > 1e-15:
            w = pair_weights[k]
            # d(pair_val_k)/d(centers[i]) = (centers[i] - centers[j]) / (2*d)
            # d(pair_val_k)/d(centers[j]) = (centers[j] - centers[i]) / (2*d)
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] += w * diff_ij / (2.0 * d)
            grad[2*j:2*j+2] -= w * diff_ij / (2.0 * d)
    
    # Boundary contributions
    bound_weights = weights[len(pair_vals):]
    for i in range(n):
        w = bound_weights[i]
        # bound_val_i = min(dist_x_i, dist_y_i)
        # If dist_x < dist_y: d/dx_i = -sign(x_i), d/dy_i = 0
        # If dist_y < dist_x: d/dx_i = 0, d/dy_i = -sign(y_i)
        # If equal: split gradient
        dx = dist_x[i]
        dy = dist_y[i]
        if dx < dy:
            gx = -np.sign(centers[i, 0]) if centers[i, 0] != 0 else 1.0
            gy = 0.0
        elif dy < dx:
            gx = 0.0
            gy = -np.sign(centers[i, 1]) if centers[i, 1] != 0 else 1.0
        else:
            gx = -0.5 * np.sign(centers[i, 0]) if centers[i, 0] != 0 else 0.5
            gy = -0.5 * np.sign(centers[i, 1]) if centers[i, 1] != 0 else 0.5
        grad[2*i] += w * gx
        grad[2*i+1] += w * gy
    
    # Final gradient is negative of d(soft_min)/dx because obj = -soft_min
    grad = -grad
    
    return obj_val, grad

def _soft_min_obj_only(x, n, beta):
    """Objective only (for scipy minimize)."""
    val, _ = _soft_min_obj_grad(x, n, beta)
    return val

def _soft_min_grad_only(x, n, beta):
    """Gradient only (for scipy minimize)."""
    _, grad = _soft_min_obj_grad(x, n, beta)
    return grad

def _smart_init(n):
    """Create a smart initial configuration."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    # Find best spot for the 50th point
    sample_res = 200
    xs = np.linspace(-0.5, 0.5, sample_res)
    ys = np.linspace(-0.5, 0.5, sample_res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    diff = pts[:, np.newaxis, :] - grid_49[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists, axis=1)
    
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    d_b = np.minimum(d_x, d_y)
    
    clearance = np.minimum(min_dists, d_b)
    idx = np.argmax(clearance)
    new_pt = pts[idx]
    
    centers = np.vstack([grid_49, new_pt])
    
    np.random.seed(42)
    centers = centers + np.random.normal(0, 0.003, size=centers.shape)
    
    return centers[:n]

def _sa_optimize(centers, n, t_start, t_end, n_iter, seed=0, time_limit=None):
    """Simulated annealing for circle packing with time limit."""
    rng = np.random.RandomState(seed)
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    start_time = time.time()
    
    for it in range(n_iter):
        # Check time limit
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Annealing schedule
        alpha = it / max(n_iter - 1, 1)
        temperature = t_start * (t_end / t_start) ** alpha
        step_size = 0.015 * (1.0 - alpha * 0.6)
        
        # Random perturbation
        new_centers = centers + rng.normal(0, step_size, size=centers.shape)
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centers)
        
        # Accept if better or with probability exp((new_r - current_r) / temperature)
        if new_r > current_r:
            centers = new_centers
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            delta = new_r - current_r
            if temperature > 1e-12:
                prob = np.exp(delta / temperature)
                if rng.random() < prob:
                    centers = new_centers
                    current_r = new_r
    
    return best_centers, best_r

def _nelder_mead_polish(centers, n, max_iter=200, tol=1e-14):
    """Nelder-Mead polish for final precision."""
    x0 = centers.flatten()
    
    def obj(x):
        c = x.reshape(n, 2)
        r = _compute_radius(c)
        return -r
    
    result = minimize(
        obj,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    
    refined_centers = result.x.reshape(n, 2)
    refined_r = _compute_radius(refined_centers)
    
    return refined_centers, refined_r

def _lbfgs_polish(centers, n, beta, max_iter=200, time_limit=None):
    """L-BFGS-B polish using soft-min with gradient."""
    x0 = centers.flatten()
    start_time = time.time()
    
    def obj(x):
        if time_limit is not None and time.time() > time_limit:
            raise RuntimeError("time limit exceeded")
        return _soft_min_obj_only(x, n, beta)
    
    def grad(x):
        if time_limit is not None and time.time() > time_limit:
            raise RuntimeError("time limit exceeded")
        return _soft_min_grad_only(x, n, beta)
    
    try:
        result = minimize(
            obj,
            x0,
            jac=grad,
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (2 * n),
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 5000}
        )
        
        refined_centers = result.x.reshape(n, 2)
        refined_r = _compute_radius(refined_centers)
        
        return refined_centers, refined_r
    except RuntimeError:
        return centers, _compute_radius(centers)
    except Exception:
        return centers, _compute_radius(centers)

def _high_precision_polish(centers, n, time_limit=None):
    """
    High-precision polish: iteratively apply L-BFGS with increasing beta,
    then Nelder-Mead for final touch.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    start_time = time.time()
    
    # Progressive beta increase
    betas = [50, 100, 200, 400, 800]
    for beta in betas:
        if time_limit is not None and time.time() > time_limit:
            break
        
        new_centers, new_r = _lbfgs_polish(current, n, beta=beta, max_iter=150, 
                                             time_limit=time_limit)
        if new_r > current_r:
            current = new_centers
            current_r = new_r
    
    # Final Nelder-Mead
    if time_limit is None or time.time() < time_limit - 2.0:
        nm_centers, nm_r = _nelder_mead_polish(current, n, max_iter=200, tol=1e-15)
        if nm_r > current_r:
            current = nm_centers
            current_r = nm_r
    
    return current, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_centers = None
    best_r = 0.0
    
    # Phase 1: Multiple SA runs with different seeds and initializations
    inits = [
        _smart_init(N),
        _smart_init(N) + np.random.RandomState(1).normal(0, 0.02, size=(N, 2)),
        _smart_init(N) + np.random.RandomState(2).normal(0, 0.04, size=(N, 2)),
        _smart_init(N) + np.random.RandomState(3).normal(0, 0.06, size=(N, 2)),
    ]
    
    sa_configs = [
        (0.04, 0.0005, 1000, 0),
        (0.03, 0.0003, 1200, 1),
        (0.02, 0.0002, 1500, 2),
    ]
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        
        init_centers = np.clip(init_centers, -0.49, 0.49)
        
        # Try SA with different configs
        for sa_idx in range(min(2, len(sa_configs))):
            if time.time() > deadline - 55.0:
                break
            
            t_start, t_end, n_iter, seed = sa_configs[sa_idx]
            
            sa_centers, sa_r = _sa_optimize(init_centers, N, t_start, t_end, n_iter, seed,
                                             time_limit=deadline - 50.0)
            
            if sa_r > best_r:
                best_r = sa_r
                best_centers = sa_centers.copy()
            
            # Quick L-BFGS polish after SA
            if time.time() < deadline - 40.0:
                polished_centers, polished_r = _lbfgs_polish(sa_centers, N, beta=100, 
                                                               max_iter=100,
                                                               time_limit=deadline - 35.0)
                if polished_r > best_r:
                    best_r = polished_r
                    best_centers = polished_centers.copy()
    
    # Phase 2: High-precision polish on best
    if best_centers is not None and time.time() < deadline - 15.0:
        hp_centers, hp_r = _high_precision_polish(best_centers, N, time_limit=deadline - 5.0)
        if hp_r > best_r:
            best_r = hp_r
            best_centers = hp_centers.copy()
    
    # Ensure validity
    if best_centers is not None:
        eps = 1e-10
        best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
        # Final check
        final_r = _compute_radius(best_centers)
        if final_r <= 0:
            best_centers = _smart_init(N)
            best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    else:
        best_centers = _smart_init(N)
        best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

# EVOLVE-BLOCK-END
