import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s limit

def _compute_vals(pts):
    """Compute the half-distances between all pairs and boundary distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Exclude self-distances (diagonal)
    d = np.where(np.eye(N, dtype=bool), np.inf, d)
    pair_half = d / 2.0
    
    # Boundary distances: 1 - |c_i|
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary = 1.0 - norms
    
    # Flatten pair half distances (excluding diagonal)
    # We only need upper triangle to avoid duplicates, but for soft-min sum, 
    # duplicates just double the weight, which is fine for the log-sum-exp form 
    # if consistent. However, to be precise and efficient, let's take unique pairs.
    i_idx, j_idx = np.triu_indices(N, k=1)
    unique_pair_halves = pair_half[i_idx, j_idx]
    
    all_vals = np.concatenate([unique_pair_halves, boundary])
    return all_vals

def _smooth_obj(x, beta):
    """
    Objective function to minimize.
    We want to maximize r = min(vals).
    soft_min(vals) = -1/beta * log(sum(exp(-beta * vals)))
    We minimize -soft_min(vals) = 1/beta * log(sum(exp(-beta * vals)))
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    vals = _compute_vals(pts)
    
    # If any val is very large negative (shouldn't happen with geometry but just in case), 
    # or if beta is large, we need numerical stability.
    # max_neg = max(-beta * vals)
    # log_sum_exp = max_neg + log(sum(exp(-beta * vals - max_neg)))
    # obj = log_sum_exp / beta
    
    neg_beta_vals = -beta * vals
    max_neg = np.max(neg_beta_vals)
    
    # If max_neg is -inf (all vals +inf?), handle it. 
    # But vals are distances, so they are positive. -beta*vals is negative.
    # max_neg is the least negative (closest to 0).
    
    exp_terms = np.exp(neg_beta_vals - max_neg)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        return 1e10
        
    log_sum_exp = max_neg + np.log(sum_exp)
    # The soft min is -log_sum_exp / beta
    # We want to maximize soft min, so minimize -soft min = log_sum_exp / beta
    return log_sum_exp / beta

def _smooth_obj_grad(x, beta):
    """
    Gradient of the smooth objective.
    f(x) = 1/beta * log(sum_k exp(-beta * v_k(x)))
    grad f = 1/beta * 1/sum_k exp(-beta * v_k) * sum_k exp(-beta * v_k) * (-beta * grad v_k)
           = - sum_k (exp(-beta * v_k) / sum_j exp(-beta * v_j)) * grad v_k
           = - sum_k w_k * grad v_k
    where w_k = exp(-beta * v_k) / sum_j exp(-beta * v_j)
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return np.zeros_like(x)
    
    # Compute values and weights
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    d = np.where(np.eye(N, dtype=bool), np.inf, d)
    pair_half = d / 2.0
    unique_pair_halves = pair_half[i_idx, j_idx]
    
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary = 1.0 - norms
    
    all_vals = np.concatenate([unique_pair_halves, boundary])
    
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    exp_terms = np.exp(neg_beta_vals - max_neg)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        return np.zeros_like(x)
        
    weights = exp_terms / sum_exp
    
    grad = np.zeros((N, 2))
    
    # Pair contributions
    # v_k = |c_i - c_j| / 2
    # grad v_k w.r.t c_i = (c_i - c_j) / |c_i - c_j|
    # grad v_k w.r.t c_j = (c_j - c_i) / |c_i - c_j|
    
    # We have unique pairs in i_idx, j_idx
    # weights[:len(i_idx)] correspond to these pairs
    w_pairs = weights[:len(i_idx)]
    
    # Calculate vectors for pairs
    c_i = pts[i_idx]
    c_j = pts[j_idx]
    vecs = c_i - c_j
    dists = np.sqrt(np.sum(vecs ** 2, axis=1))
    
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    inv_dists = 1.0 / safe_dists
    
    # Gradient of distance/2 w.r.t c_i is vecs * inv_dists
    # Gradient of distance/2 w.r.t c_j is -vecs * inv_dists
    
    # We need to scatter-add these gradients
    # grad_c_i += w * (c_i - c_j) / d
    # grad_c_j += w * (c_j - c_i) / d
    
    # Using np.add.at for scatter addition
    np.add.at(grad, i_idx, w_pairs[:, np.newaxis] * vecs * inv_dists[:, np.newaxis])
    np.add.at(grad, j_idx, -w_pairs[:, np.newaxis] * vecs * inv_dists[:, np.newaxis])
    
    # Boundary contributions
    # v_k = 1 - |c_i|
    # grad v_k w.r.t c_i = -c_i / |c_i|
    
    w_bound = weights[len(i_idx):]
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    bound_grads = -w_bound[:, np.newaxis] * pts / safe_norms[:, np.newaxis]
    
    grad += bound_grads
    
    # The objective is -soft_min, so we return the gradient of the function to minimize.
    # f(x) = log_sum_exp / beta
    # grad f = - sum w_k grad v_k
    # So we return - (pair_grad + bound_grad)
    # Wait, my derivation:
    # f = 1/beta log sum exp(-beta v)
    # df/dx = 1/beta * 1/S * sum exp(-beta v) * (-beta dv/dx) = - 1/S sum exp(-beta v) dv/dx = - sum w_k dv/dx
    # So the gradient of the function to minimize is - sum w_k grad v_k.
    # My `grad` variable currently holds sum w_k grad v_k.
    # So I should return -grad.
    
    return (-grad).ravel()

def construct():
    global START_TIME, TIME_LIMIT
    
    # Initial guess: Random points within radius 0.5
    np.random.seed(42)
    x0 = np.random.uniform(-0.5, 0.5, size=(N, 2)).ravel()
    
    best_x = x0.copy()
    best_val = 1e10 # We are minimizing, so start high
    
    # Multi-stage beta optimization
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Optimize with current beta
        # Use L-BFGS-B as it handles gradients well
        try:
            res = minimize(
                _smooth_obj, 
                current_x, 
                jac=_smooth_obj_grad, 
                method='L-BFGS-B', 
                args=(beta,),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_val:
                current_x = res.x
                # Evaluate the actual hard min to track best
                pts = current_x.reshape(N, 2)
                vals = _compute_vals(pts)
                actual_r = np.min(vals)
                # Since we minimize -soft_min, lower res.fun is better.
                # But we want to track the best hard radius.
                if not np.isfinite(best_val) or actual_r > best_val:
                    best_val = actual_r
                    best_x = current_x.copy()
        except Exception:
            pass
            
    # Final polish: Use the best found x and try to improve with a very high beta
    # or just return the best found.
    # Let's do one more optimization with the best x found so far, using a high beta
    if time.time() - START_TIME < TIME_LIMIT - 5:
        try:
            # Use the best_x as start
            res = minimize(
                _smooth_obj, 
                best_x, 
                jac=_smooth_obj_grad, 
                method='L-BFGS-B', 
                args=(5000,),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < 1e10: # Check if valid
                pts = res.x.reshape(N, 2)
                vals = _compute_vals(pts)
                actual_r = np.min(vals)
                if actual_r > best_val:
                    best_x = res.x.copy()
                    best_val = actual_r
        except Exception:
            pass

    # Return the best configuration found
    return best_x.reshape(N, 2)
