"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 140  # 10s safety margin from 150s limit

def _hex_lattice_init():
    """Generate initial positions using a hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Select N points closest to center to keep them well-distributed
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        spacing *= 0.98

def _calc_score(C):
    """Calculate the effective radius for a given set of centers."""
    C = np.asarray(C).reshape(-1, 2)
    if C.shape[0] != N:
        return 0.0
    # Check for non-finite or out-of-bound
    if not np.all(np.isfinite(C)):
        return 0.0
    
    # Distance to boundary
    norms = np.linalg.norm(C, axis=1)
    r_bound = 1.0 - np.max(norms)
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r_pair = min_dist / 2.0
    
    return min(r_bound, r_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Smooth surrogate for maximizing min(r_bound, r_pair) using soft-min.
    We minimize -softmin where softmin approximates the minimum of constraint values.
    """
    C = x.reshape(N, 2)
    
    # Constraint values (larger is better):
    # g_i = 1 - ||c_i|| for boundary
    # h_ij = ||c_i - c_j|| / 2 for pairwise
    
    norms = np.linalg.norm(C, axis=1)
    g_boundary = 1.0 - norms  # shape (N,)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    h_pairs = dists[i_idx, j_idx] / 2.0  # shape (N*(N-1)/2,)
    
    # All constraint values
    all_vals = np.concatenate([g_boundary, h_pairs])
    
    # Soft-min: softmin(v) = -logsumexp(-beta * v) / beta
    # To avoid overflow, use log-sum-exp trick
    neg_beta_v = -beta * all_vals
    max_val = np.max(neg_beta_v)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_v - max_val)))
    softmin_val = -log_sum_exp / beta
    
    # Objective: minimize -softmin
    obj = -softmin_val
    
    # Gradient of -softmin w.r.t. x
    # d(softmin)/dx = (1/beta) * (sum(exp(-beta*v_k) * (-beta) * dv_k/dx) / sum(exp(-beta*v_k)))
    # = -sum(exp(-beta*v_k) * dv_k/dx) / sum(exp(-beta*v_k))
    # d(-softmin)/dx = sum(exp(-beta*v_k) * dv_k/dx) / sum(exp(-beta*v_k))
    
    exp_vals = np.exp(neg_beta_v - max_val)  # normalized (before dividing by sum)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # shape (N + N*(N-1)/2,)
    
    # Gradient of each constraint value w.r.t. x
    # d g_i / d c_i = -c_i / ||c_i||
    # d h_ij / d c_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d h_ij / d c_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    grad = np.zeros_like(C)
    
    # Boundary part
    safe_norms = np.where(norms > 1e-10, norms, 1e-10)
    grad_boundary = -C / safe_norms[:, np.newaxis]  # shape (N, 2)
    
    # Weighted sum for boundary
    w_boundary = weights[:N]  # shape (N,)
    grad += (w_boundary[:, np.newaxis] * grad_boundary).sum(axis=0)
    
    # Pairwise part
    w_pairs = weights[N:]  # shape (N*(N-1)/2,)
    valid_d = dists[i_idx, j_idx]
    safe_d = np.where(valid_d > 1e-10, valid_d, 1e-10)
    
    # d h_ij / d c_i = (c_i - c_j) / (2 * d)
    # d h_ij / d c_j = (c_j - c_i) / (2 * d)
    
    d_vec = C[i_idx] - C[j_idx]  # shape (pairs, 2)
    factor_pairs = w_pairs / (2.0 * safe_d)  # shape (pairs,)
    
    # Accumulate gradients
    # For each pair, add factor * d_vec to grad[i] and -factor * d_vec to grad[j]
    # Use np.add.at for accumulation
    grad_i = factor_pairs[:, np.newaxis] * d_vec
    grad_j = -factor_pairs[:, np.newaxis] * d_vec
    
    np.add.at(grad, i_idx, grad_i)
    np.add.at(grad, j_idx, grad_j)
    
    return obj, grad.flatten()

def _objective_and_grad(x, r_target):
    """
    Calculate the violation cost and gradient for a given radius target.
    x: flattened array of centers (2*N)
    r_target: target radius
    """
    C = x.reshape(N, 2)
    
    # 1. Boundary Violations
    norms = np.linalg.norm(C, axis=1)
    # Violation if norm > 1 - r_target
    # v_i = max(0, norm_i - (1 - r_target))
    # Cost = sum v_i^2
    max_norm_allowed = 1.0 - r_target
    over_bound = norms - max_norm_allowed
    pos_over_bound = np.maximum(0.0, over_bound)
    
    cost_bound = np.sum(pos_over_bound ** 2)
    
    # Gradient for boundary part
    # d/dC_i sum(max(0, ||C_i|| - M)^2)
    # = 2 * max(0, ||C_i|| - M) * (C_i / ||C_i||)
    # If ||C_i|| is 0, gradient is 0 (or undefined, treat as 0)
    grad_bound = np.zeros_like(C)
    mask = pos_over_bound > 0
    if np.any(mask):
        # C[mask] / norms[mask]
        # Avoid division by zero
        safe_norms = norms[mask]
        safe_norms[safe_norms == 0] = 1e-10
        grad_bound[mask] = 2.0 * pos_over_bound[mask, np.newaxis] * (C[mask] / safe_norms[:, np.newaxis])

    # 2. Pairwise Distance Violations
    # For i < j, d_ij < 2 * r_target is a violation
    # v_ij = max(0, 2*r_target - d_ij)
    # Cost = sum v_ij^2
    
    # Compute pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    min_dist_req = 2.0 * r_target
    # We only care about i < j to avoid double counting in sum, 
    # but for gradient we need to account for both i and j contributions.
    # Sum over all i != j of max(0, R - d_ij)^2 / 2 ? 
    # The objective is sum_{i<j} v_ij^2.
    # Let's compute the upper triangle.
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs = dists[i_idx, j_idx]
    
    over_dist = min_dist_req - d_pairs
    pos_over_dist = np.maximum(0.0, over_dist)
    
    cost_pair = np.sum(pos_over_dist ** 2)
    
    # Gradient for pairwise part
    # Term for pair (i,j): v^2 = (R - d)^2
    # d/dc_i v^2 = 2 * v * (-(c_i - c_j) / d)
    # d/dc_j v^2 = 2 * v * (-(c_j - c_i) / d)
    
    grad_pair = np.zeros_like(C)
    mask_pair = pos_over_dist > 0
    if np.any(mask_pair):
        valid_i = i_idx[mask_pair]
        valid_j = j_idx[mask_pair]
        valid_d = d_pairs[mask_pair]
        valid_v = pos_over_dist[mask_pair]
        
        # Avoid division by zero
        valid_d[valid_d == 0] = 1e-10
        
        # Gradient vector for each pair
        # d_vec = c_i - c_j
        d_vec = C[valid_i] - C[valid_j]
        
        # Factor: 2 * v / d
        factor = (2.0 * valid_v / valid_d)
        
        # Grad for i: -factor * d_vec
        # Grad for j: +factor * d_vec
        
        # Vectorized accumulation using np.add.at
        grad_i_contrib = -factor[:, np.newaxis] * d_vec
        grad_j_contrib = factor[:, np.newaxis] * d_vec
        
        np.add.at(grad_pair, valid_i, grad_i_contrib)
        np.add.at(grad_pair, valid_j, grad_j_contrib)

    total_cost = cost_bound + cost_pair
    total_grad = (grad_bound + grad_pair).flatten()
    
    return total_cost, total_grad

def _polish(C_start, max_iter=5000):
    """
    Final high-precision polish pass using soft-min with increasing beta.
    """
    C = C_start.copy()
    beta_values = [50.0, 100.0, 200.0, 500.0]
    
    for beta in beta_values:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        def fun(x):
            return _soft_min_objective_and_grad(x, beta)[0]
        
        def jac(x):
            return _soft_min_objective_and_grad(x, beta)[1]
        
        try:
            res = minimize(
                fun, 
                C.flatten(), 
                jac=jac, 
                method='L-BFGS-B', 
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            C_new = res.x.reshape(N, 2)
            # Verify no degradation
            if _calc_score(C_new) >= _calc_score(C) - 1e-12:
                C = C_new
        except Exception:
            pass
    
    return C

def construct():
    # Initial guess from hex lattice
    C_init = _hex_lattice_init()
    best_C = C_init.copy()
    best_score = _calc_score(best_C)
    
    # Target radius search
    # Known best is ~0.125825
    # Start slightly below to ensure feasibility, then push up
    r_low = best_score
    r_high = 0.1260  # Slightly above known best
    
    # If best_score is already high, narrow the range
    if best_score > 0.125:
        r_low = best_score
        r_high = best_score + 0.001

    # Binary search for the maximum feasible radius
    # We will run the optimizer for each candidate radius
    # and check if the resulting configuration is actually valid (score >= r_target - epsilon)
    
    num_iters = 8  # Limit iterations to fit in time
    
    for _ in range(num_iters):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Objective function for minimize
        def fun(x):
            return _objective_and_grad(x, r_mid)[0]
        
        def jac(x):
            return _objective_and_grad(x, r_mid)[1]
        
        try:
            res = minimize(
                fun, 
                best_C.flatten(), 
                jac=jac, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            C_new = res.x.reshape(N, 2)
            score_new = _calc_score(C_new)
            
            # Update best
            if score_new > best_score:
                best_score = score_new
                best_C = C_new.copy()
            
            # Check if feasible for r_mid
            # If score_new >= r_mid (or close), it's feasible
            if score_new >= r_mid - 1e-5:
                r_low = r_mid
            else:
                r_high = r_mid
                
        except Exception:
            break
    
    # Final polish pass with soft-min
    if time.time() - START_TIME < TIME_LIMIT - 10:
        C_polished = _polish(best_C)
        score_polished = _calc_score(C_polished)
        if score_polished > best_score:
            best_score = score_polished
            best_C = C_polished
    
    return best_C

# EVOLVE-BLOCK-END
