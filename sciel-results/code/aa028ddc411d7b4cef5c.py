import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_max_min_objective(params, beta=10.0):
    """
    Smooth objective: minimize soft_max(d_sq) / soft_min(d_sq)
    using log-sum-exp for numerical stability.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    # Soft max: (1/beta) * log(sum(exp(beta * d_sq)))
    max_d = np.max(d_sq)
    exp_vals = np.exp(beta * (d_sq - max_d))
    soft_max = max_d + np.log(np.sum(exp_vals)) / beta
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * d_sq)))
    min_d = np.min(d_sq)
    exp_vals_min = np.exp(-beta * (d_sq - min_d))
    soft_min = min_d - np.log(np.sum(exp_vals_min)) / beta
    
    if soft_min < 1e-6:
        return 1e10
    
    return soft_max / soft_min

def _soft_max_min_gradient(params, beta=10.0):
    """
    Analytical gradient of soft_max(d_sq) / soft_min(d_sq) w.r.t. params.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    # Compute pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq_full = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq = d_sq_full[iu]
    n_pairs = len(d_sq)
    
    # Soft max computation
    max_d = np.max(d_sq)
    exp_max = np.exp(beta * (d_sq - max_d))
    sum_exp_max = np.sum(exp_max)
    w_max = exp_max / sum_exp_max  # weights for soft max
    soft_max = max_d + np.log(sum_exp_max) / beta
    
    # Soft min computation
    min_d = np.min(d_sq)
    exp_min = np.exp(-beta * (d_sq - min_d))
    sum_exp_min = np.sum(exp_min)
    w_min = exp_min / sum_exp_min  # weights for soft min
    soft_min = min_d - np.log(sum_exp_min) / beta
    
    if soft_min < 1e-6:
        return np.zeros_like(params)
    
    # d(soft_max)/d(d_sq[i]) = w_max[i]
    # d(soft_min)/d(d_sq[i]) = -w_min[i]
    # 
    # f = soft_max / soft_min
    # df/d(d_sq[i]) = (w_max[i] * soft_min - soft_max * (-w_min[i])) / soft_min^2
    #               = (w_max[i] * soft_min + soft_max * w_min[i]) / soft_min^2
    
    grad_d_sq = (w_max * soft_min + soft_max * w_min) / (soft_min ** 2)
    
    # Now chain rule: d(d_sq[i])/d(params)
    # d_sq[i] = |P[a] - P[b]|^2 for pair (a,b)
    # d(d_sq[i])/d(P[k]) = 2*(P[k] - P[j]) if k==a, 2*(P[k] - P[i]) if k==b, 0 otherwise
    
    # Build index mapping from pair index to (a, b)
    pair_a, pair_b = iu
    
    # For each parameter (point k, coord c), accumulate gradient
    # params maps to P[2:], so param index j corresponds to P[j+2]
    grad_params = np.zeros_like(params)
    
    for i in range(n_pairs):
        a = pair_a[i]
        b = pair_b[i]
        g = grad_d_sq[i]
        
        # Contribution to point a
        if a >= 2:
            grad_params[(a-2)*2:(a-2)*2+2] += 2 * g * (P[a] - P[b])
        # Contribution to point b
        if b >= 2:
            grad_params[(b-2)*2:(b-2)*2+2] += 2 * g * (P[b] - P[a])
        
        # Contributions to fixed points 0 and 1 don't affect params
    
    return grad_params

def _exact_objective(params):
    """Exact objective for local refinement."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Hexagonal Lattice Subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    hex_points = hex_points[sorted_indices]
    candidates.append(hex_points[:16])
    
    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles (8 inner, 8 outer)
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([1.0 * np.cos(ang), 1.0 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([0.5 * np.cos(ang), 0.5 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 4. Random Perturbed Grid
    perturbed = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed)
    
    # 5. Random Gaussian Cluster
    gauss = rng.normal(0, 1, (N, 2))
    candidates.append(gauss)
    
    # 6. Ring of 16
    pts_ring = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_ring.append([1.0 * np.cos(ang), 1.0 * np.sin(ang)])
    candidates.append(np.array(pts_ring))
    
    # 7. Two rings: 8 and 8, different radii
    pts_2rings = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_2rings.append([1.5 * np.cos(ang), 1.5 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_2rings.append([0.5 * np.cos(ang), 0.5 * np.sin(ang)])
    candidates.append(np.array(pts_2rings))
    
    # 8. Fibonacci sphere-like distribution projected to 2D (golden angle)
    pts_fib = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for k in range(16):
        r = np.sqrt(k / 15.0)
        ang = golden_angle * k
        pts_fib.append([r * np.cos(ang), r * np.sin(ang)])
    candidates.append(np.array(pts_fib))
    
    # 9. Three concentric rings: 6, 5, 5
    pts_3rings = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts_3rings.append([0.4 * np.cos(ang), 0.4 * np.sin(ang)])
    for k in range(5):
        ang = 2 * np.pi * (k + 0.5) / 5
        pts_3rings.append([0.8 * np.cos(ang), 0.8 * np.sin(ang)])
    for k in range(5):
        ang = 2 * np.pi * k / 5
        pts_3rings.append([1.2 * np.cos(ang), 1.2 * np.sin(ang)])
    candidates.append(np.array(pts_3rings))
    
    # 10. Random with moderate spread
    for seed in range(3):
        rng_local = np.random.default_rng(seed + 100)
        pts_rand = rng_local.uniform(-1.5, 1.5, (N, 2))
        candidates.append(pts_rand)
    
    return candidates

def _prepare_for_optimization(P):
    """Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P."""
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    p, q = np.unravel_index(min_idx, (N, N))
    if p > q:
        p, q = q, p
    
    p0 = P[p]
    p1 = P[q]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return None, None
        
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (P - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    return init_params, perm

def _local_optimize(params, deadline, beta_schedule=None):
    """
    Multi-stage local optimization:
    1. Coarse-to-fine with decreasing beta
    2. Final polish with exact objective
    """
    if beta_schedule is None:
        beta_schedule = [5.0, 10.0, 20.0, 40.0]
    
    current_params = params.copy()
    best_score = _exact_objective(current_params)
    best_params = current_params.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _soft_max_min_objective,
                current_params,
                jac=lambda x: _soft_max_min_gradient(x, beta=beta),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if res.success or res.fun < best_score:
                P_check = np.zeros((N, 2))
                P_check[0] = [0.0, 0.0]
                P_check[1] = [1.0, 0.0]
                P_check[2:] = res.x.reshape(-1, 2)
                s = _score(P_check)
                if s < best_score:
                    best_score = s
                    best_params = res.x.copy()
                    current_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on exact objective
    if time.time() < deadline - 3:
        try:
            res_final = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            P_final = np.zeros((N, 2))
            P_final[0] = [0.0, 0.0]
            P_final[1] = [1.0, 0.0]
            P_final[2:] = res_final.x.reshape(-1, 2)
            
            final_score = _score(P_final)
            if final_score < best_score:
                best_score = final_score
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    
    results = []
    
    for i, P0 in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        init_params, perm = _prepare_for_optimization(P0)
        if init_params is None:
            continue
        
        try:
            score, params = _local_optimize(init_params, deadline)
            
            if score < best_score:
                best_score = score
                best_params = params
                
            results.append((score, params))
            
        except Exception:
            pass
    
    # If we have time, do additional random restarts from best solution
    if best_params is not None and time.time() < deadline - 15:
        rng = np.random.default_rng(123)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            perturbation_scale = 0.05 * (restart + 1)
            perturbed = best_params + rng.normal(0, perturbation_scale, best_params.shape)
            
            try:
                score, params = _local_optimize(perturbed, deadline, beta_schedule=[10.0, 20.0, 40.0])
                if score < best_score:
                    best_score = score
                    best_params = params
            except Exception:
                pass
    
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        if _score(final_pts) > 1e6:
            return candidates[0]
            
        return final_pts
    else:
        return candidates[0]

# EVOLVE-BLOCK-END
