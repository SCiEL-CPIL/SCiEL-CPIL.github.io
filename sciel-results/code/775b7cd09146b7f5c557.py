"""Improved program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _softmin_logsumexp(x, beta):
    """Numerically stable soft-min using log-sum-exp."""
    x_min = np.min(x)
    exp_terms = np.exp(-beta * (x - x_min))
    return x_min - np.log(np.sum(exp_terms)) / beta

def _energy_smooth(P, r, beta=50.0):
    """
    Smooth energy using soft-min for better gradient flow.
    Penalizes violations of circle-circle and circle-boundary constraints.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Circle-circle: want dist >= 2r, penalize when dist < 2r
    cc_violation = 2.0 * r - dist
    # Use soft-max to get smooth approximation of max(0, violation)
    cc_penalty = _softmin_logsumexp(-cc_violation, beta) / beta
    
    # Circle-boundary: want |P| + r <= 1, i.e., |P| <= 1 - r
    dist_origin = np.linalg.norm(P, axis=1)
    cb_violation = dist_origin - (1.0 - r)
    cb_penalty = _softmin_logsumexp(-cb_violation, beta) / beta
    
    return cc_penalty + cb_penalty

def _energy_smooth_grad(P, r, beta=50.0):
    """
    Gradient of the smooth energy function.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Circle-circle gradient
    cc_violation = 2.0 * r - dist
    # Soft-max weights: w_ij = exp(-beta * (-cc_violation_ij)) / sum(exp(...))
    # = exp(beta * cc_violation_ij) / sum(exp(beta * cc_violation))
    # But we need to handle this carefully for numerical stability
    
    # For each pair (i,j), the contribution to grad_i is proportional to:
    # (P_i - P_j) / dist_ij * d(softmax)/d(cc_violation_ij)
    # d(softmax)/dx = exp(beta*x) / sum(exp(beta*x))
    
    # Compute weights for circle-circle
    cc_violation_safe = np.where(dist > 0, cc_violation, -np.inf)
    exp_cc = np.exp(beta * cc_violation_safe)
    np.fill_diagonal(exp_cc, 0.0)
    sum_exp_cc = np.sum(exp_cc, axis=1, keepdims=True)
    sum_exp_cc = np.where(sum_exp_cc > 0, sum_exp_cc, 1.0)
    w_cc = exp_cc / sum_exp_cc
    
    # grad_i += sum_j w_cc[i,j] * (P_i - P_j) / dist_ij
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    np.fill_diagonal(inv_dist, 0.0)
    
    # term1[i] = sum_j w_cc[i,j] * P_j / dist_ij
    term1_cc = (w_cc * inv_dist) @ P
    # term2[i] = P_i * sum_j w_cc[i,j] / dist_ij
    term2_cc = P * np.sum(w_cc * inv_dist, axis=1)[:, np.newaxis]
    grad += term2_cc - term1_cc
    
    # Circle-boundary gradient
    dist_origin = np.linalg.norm(P, axis=1)
    cb_violation = dist_origin - (1.0 - r)
    exp_cb = np.exp(beta * cb_violation)
    sum_exp_cb = np.sum(exp_cb)
    sum_exp_cb = max(sum_exp_cb, 1e-30)
    w_cb = exp_cb / sum_exp_cb
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    # grad_i += w_cb[i] * P_i / dist_origin[i]
    grad += (w_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    radii = params[:k]
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
            
    return P

def _optimize_positions(P_init, r_target, time_limit=5.0, beta=50.0):
    """
    Optimize positions P to minimize smooth energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_smooth(P, r_target, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_smooth_grad(P, r_target, beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())

def _global_search_ring_params(ring_counts, time_limit=30.0):
    """
    Use differential evolution to find good ring radii and offsets.
    """
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_initialization(params, ring_counts)
        score = _compute_score(P)
        return -score
        
    start_time = time.time()
    
    try:
        res = differential_evolution(neg_score, bounds, maxiter=50, popsize=15, 
                                     tol=1e-5, seed=42, polish=True, 
                                     workers=1,
                                     init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        
    except Exception:
        best_P = _get_hex_initial_points()
        
    return best_P

def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98

def _local_refine(P_init, r_init, deadline):
    """
    Local refinement using smooth energy gradient-based optimization.
    """
    P_curr = P_init.copy()
    r_curr = r_init
    
    # Progressive beta annealing for smoother optimization
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        def objective(P_flat):
            P = P_flat.reshape(N, 2)
            return _energy_smooth(P, r_curr, beta)
        
        def grad(P_flat):
            P = P_flat.reshape(N, 2)
            return _energy_smooth_grad(P, r_curr, beta).flatten()
        
        try:
            result = minimize(objective, P_curr.flatten(), jac=grad, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
        except Exception:
            pass
    
    # Final high-precision polish with very high beta
    if time.time() < deadline - 5.0:
        beta_final = 500.0
        def objective(P_flat):
            P = P_flat.reshape(N, 2)
            return _energy_smooth(P, r_curr, beta_final)
        
        def grad(P_flat):
            P = P_flat.reshape(N, 2)
            return _energy_smooth_grad(P, r_curr, beta_final).flatten()
        
        try:
            result = minimize(objective, P_curr.flatten(), jac=grad, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
        except Exception:
            pass
    
    # Manual refinement pass for fine-tuning
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
            
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        forces = np.zeros_like(P_curr)
        
        overlap_cc = np.maximum(2.0 * r_curr - dist, 0.0)
        mask_cc = overlap_cc > 0
        if np.any(mask_cc):
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 0)
            weight = np.zeros((N, N))
            weight[mask_cc] = overlap_cc[mask_cc] * inv_dist[mask_cc]
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            forces += 2.0 * (term1 - term2)
        
        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_curr - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]
        
        norm_forces = np.linalg.norm(forces)
        if norm_forces < 1e-15:
            break
        
        best_step = 0
        best_score = r_curr
        for factor in [0.05, 0.02, 0.01, 0.005, 0.002]:
            step = factor * norm_forces / (N * N + 1)
            P_new = P_curr + step * forces
            new_score = _compute_score(P_new)
            if new_score > best_score:
                best_score = new_score
                best_step = step
        
        if best_step > 0:
            P_curr = P_curr + best_step * forces
            r_curr = best_score
        else:
            break
    
    return P_curr, r_curr

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer
    
    np.random.seed(42)
    
    # Define ring structures
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16]
    ]
    
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    for ring_counts in ring_structures:
        if time.time() > deadline - 40.0:
            break
            
        # 1. Global search for ring parameters
        P_candidate = _global_search_ring_params(ring_counts, time_limit=20.0)
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Local refinement with binary search on r
        r_low = score_best * 0.9
        r_high = 0.15
        
        P_opt, E = _optimize_positions(P_candidate, r_low, time_limit=5.0)
        if E < 1e-6:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
            r_low = score
            
        # Binary search with adaptive beta
        for _ in range(10):
            if time.time() > deadline - 15.0:
                break
                
            r_mid = (r_low + r_high) / 2.0
            
            P_init = P_best.copy()
            
            P_opt, E = _optimize_positions(P_init, r_mid, time_limit=5.0, beta=100.0)
            if E < 1e-6:
                r_low = r_mid
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt
                    score_best = score
            else:
                r_high = r_mid
        
        # Final local refinement on the best found so far
        P_ref, score_ref = _local_refine(P_best, score_best, deadline)
        if score_ref > score_best:
            P_best = P_ref
            score_best = score_ref
    
    # Final safety check and high-precision polish
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 3.0:
        P_best, score_best = _local_refine(P_best, score_best, deadline)
    
    final_score = _compute_score(P_best)
    if not np.isfinite(final_score) or final_score <= 0:
        P_best = _get_hex_initial_points()
    
    return P_best
# EVOLVE-BLOCK-END
