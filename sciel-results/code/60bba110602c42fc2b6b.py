"""Packing 300 equal circles in a unit circle using soft-min L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 115.0  # safety margin before 150s kill


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        return _hex_init_fallback(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _hex_init_fallback(n=N):
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _soft_min(x, beta):
    """Soft minimum of array x using log-sum-exp trick. Lower beta = sharper min."""
    x_min = np.min(x)
    # For numerical stability, subtract min
    x_shifted = x - x_min
    # Clip to avoid overflow in exp
    x_shifted = np.clip(x_shifted, 0, 50.0 / beta)
    log_sum_exp = np.log(np.sum(np.exp(-beta * x_shifted)))
    return x_min - log_sum_exp / beta


def _objective_and_grad(x_flat, beta):
    """
    Objective: negative of soft-min of (min pair distance / 2, boundary margins).
    We want to maximize r, so minimize -r.
    
    r = min(min_{i<j} d_ij/2, min_i (1 - |c_i|))
    
    We use soft-min to make it differentiable.
    """
    pts = x_flat.reshape(-1, 2)
    n = pts.shape[0]
    
    # Compute all pairwise distances
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 1e-12)  # avoid sqrt of negative
    
    # Pair distances / 2 (these are the "radii" from pairwise constraint)
    pair_r = np.sqrt(d2) / 2.0
    pair_r_flat = pair_r[np.triu_indices(n, k=1)]
    
    # Boundary margins: 1 - |c_i|
    rad = np.hypot(pts[:, 0], pts[:, 1])
    bound_r = 1.0 - rad
    
    # Combine all constraints into one array
    all_r = np.concatenate([pair_r_flat, bound_r])
    
    # Soft minimum
    r_soft = _soft_min(all_r, beta)
    
    # Objective is -r_soft (we minimize this)
    obj = -r_soft
    
    # Gradient computation
    # d r_soft / d x_i for each element in all_r
    # r_soft = x_min - (1/beta) * log(sum(exp(-beta * (x - x_min))))
    # Let s_i = exp(-beta * (x_i - x_min)) / sum_j exp(-beta * (x_j - x_min))
    # Then r_soft = x_min - (1/beta) * log(sum(s_i) * sum(exp(...))) ... 
    # Actually: r_soft = x_min - (1/beta) * log(sum_k exp(-beta*(x_k - x_min)))
    #           = x_min - (1/beta) * log(sum_k exp(-beta*x_k + beta*x_min))
    #           = x_min - (1/beta) * (log(sum_k exp(-beta*x_k)) + beta*x_min)
    #           = x_min - x_min - (1/beta)*log(sum_k exp(-beta*x_k))
    #           = -(1/beta) * log(sum_k exp(-beta*x_k))
    # 
    # d r_soft / d x_k = -(1/beta) * (1/sum(exp(-beta*x))) * (-beta * exp(-beta*x_k))
    #                  = exp(-beta*x_k) / sum_j exp(-beta*x_j)
    # 
    # So weight_k = exp(-beta*x_k) / sum_j exp(-beta*x_j)
    
    x_min = np.min(all_r)
    x_shifted = all_r - x_min
    x_shifted = np.clip(x_shifted, 0, 50.0 / beta)
    exp_terms = np.exp(-beta * x_shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # shape: (n_pairs + n,)
    
    # Gradient of r_soft w.r.t. x_flat
    grad_r = np.zeros_like(x_flat)
    
    # Contribution from pairwise constraints
    # pair_r[k] = sqrt(d2[i,j])/2 for pair (i,j)
    # d pair_r / d pts[i] = (pts[i] - pts[j]) / (2 * sqrt(d2[i,j])) = (pts[i]-pts[j]) / (4 * pair_r[k])
    # Actually: d (sqrt(d2)/2) / d pts[i] = (1/2) * (1/(2*sqrt(d2))) * 2*(pts[i]-pts[j])
    #                              = (pts[i]-pts[j]) / (2*sqrt(d2))
    #                              = (pts[i]-pts[j]) / (4 * (sqrt(d2)/2))
    # Let's recompute: let d = sqrt(d2), r_pair = d/2
    # d r_pair / d pts[i] = (1/2) * d d / d pts[i] = (1/2) * (pts[i]-pts[j])/d = (pts[i]-pts[j])/(2*d)
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    pair_count = len(idx_i)
    
    for k in range(pair_count):
        i, j = idx_i[k], idx_j[k]
        w = weights[k]  # weight for this pair constraint
        if w < 1e-15:
            continue
        d_val = np.sqrt(d2[i, j])
        if d_val < 1e-12:
            continue
        # d r_pair / d pts[i] = (pts[i] - pts[j]) / (2 * d_val)
        # d r_pair / d pts[j] = (pts[j] - pts[i]) / (2 * d_val)
        diff = (pts[i] - pts[j]) / (2.0 * d_val)
        grad_r[2*i] += w * diff[0]
        grad_r[2*i+1] += w * diff[1]
        grad_r[2*j] += w * (-diff[0])
        grad_r[2*j+1] += w * (-diff[1])
    
    # Contribution from boundary constraints
    # bound_r[i] = 1 - |c_i|
    # d bound_r / d pts[i] = -pts[i] / |c_i|
    for i in range(n):
        w = weights[pair_count + i]
        if w < 1e-15:
            continue
        r_i = rad[i]
        if r_i < 1e-12:
            continue
        # d bound_r / d pts[i] = -pts[i] / r_i
        grad_r[2*i] += w * (-pts[i, 0] / r_i)
        grad_r[2*i+1] += w * (-pts[i, 1] / r_i)
    
    # Objective is -r_soft, so gradient is -grad_r
    grad_obj = -grad_r
    
    return obj, grad_obj


def _objective_only(x_flat, beta):
    """Objective only (for L-BFGS-B when gradient is expensive)."""
    pts = x_flat.reshape(-1, 2)
    n = pts.shape[0]
    
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 1e-12)
    
    pair_r = np.sqrt(d2) / 2.0
    pair_r_flat = pair_r[np.triu_indices(n, k=1)]
    
    rad = np.hypot(pts[:, 0], pts[:, 1])
    bound_r = 1.0 - rad
    
    all_r = np.concatenate([pair_r_flat, bound_r])
    r_soft = _soft_min(all_r, beta)
    return -r_soft


def _grad_only(x_flat, beta):
    """Gradient only."""
    pts = x_flat.reshape(-1, 2)
    n = pts.shape[0]
    
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 1e-12)
    
    pair_r = np.sqrt(d2) / 2.0
    pair_r_flat = pair_r[np.triu_indices(n, k=1)]
    
    rad = np.hypot(pts[:, 0], pts[:, 1])
    bound_r = 1.0 - rad
    
    all_r = np.concatenate([pair_r_flat, bound_r])
    
    x_min = np.min(all_r)
    x_shifted = all_r - x_min
    x_shifted = np.clip(x_shifted, 0, 50.0 / beta)
    exp_terms = np.exp(-beta * x_shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    grad_r = np.zeros_like(x_flat)
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    pair_count = len(idx_i)
    
    for k in range(pair_count):
        i, j = idx_i[k], idx_j[k]
        w = weights[k]
        if w < 1e-15:
            continue
        d_val = np.sqrt(d2[i, j])
        if d_val < 1e-12:
            continue
        diff = (pts[i] - pts[j]) / (2.0 * d_val)
        grad_r[2*i] += w * diff[0]
        grad_r[2*i+1] += w * diff[1]
        grad_r[2*j] += w * (-diff[0])
        grad_r[2*j+1] += w * (-diff[1])
    
    for i in range(n):
        w = weights[pair_count + i]
        if w < 1e-15:
            continue
        r_i = rad[i]
        if r_i < 1e-12:
            continue
        grad_r[2*i] += w * (-pts[i, 0] / r_i)
        grad_r[2*i+1] += w * (-pts[i, 1] / r_i)
    
    return -grad_r


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    
    # Ensure valid start
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Annealing schedule: start with low beta (smooth), increase to sharp min
    # beta controls sharpness of soft-min: higher beta -> closer to true min
    betas = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        x0 = best_pts.flatten().copy()
        
        # Use L-BFGS-B with analytical gradient
        def _obj_grad(x_flat):
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutError()
            return _objective_and_grad(x_flat, beta)
        
        try:
            result = minimize(
                _obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'maxfun': 300,
                    'ftol': 1e-10,
                    'gtol': 1e-6,
                    'disp': False,
                }
            )
            
            if result.x is not None and np.all(np.isfinite(result.x)):
                new_pts = result.x.reshape(N, 2)
                rad_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
                # Clip to unit circle
                out_new = rad_new > 1.0
                if np.any(out_new):
                    new_pts[out_new] = new_pts[out_new] / rad_new[out_new, None]
                
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    
        except Exception:
            pass
    
    # Final refinement with very high beta if time permits
    if time.time() - start_time < TIME_LIMIT - 15.0:
        beta_final = 1000.0
        x0 = best_pts.flatten().copy()
        
        try:
            def _obj_grad_final(x_flat):
                if time.time() - start_time > TIME_LIMIT - 5.0:
                    raise TimeoutError()
                return _objective_and_grad(x_flat, beta_final)
            
            result_final = minimize(
                _obj_grad_final,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'maxfun': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'disp': False,
                }
            )
            
            if result_final.x is not None and np.all(np.isfinite(result_final.x)):
                new_pts = result_final.x.reshape(N, 2)
                rad_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
                out_new = rad_new > 1.0
                if np.any(out_new):
                    new_pts[out_new] = new_pts[out_new] / rad_new[out_new, None]
                
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    
        except Exception:
            pass
    
    # Final safety checks
    best_pts = np.asarray(best_pts, dtype=np.float64)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
    
    return best_pts
