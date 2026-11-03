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
    """Numerically stable soft-min via log-sum-exp."""
    x_min = np.min(x)
    if not np.isfinite(x_min):
        return x_min
    shifted = x - x_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return x_min - log_sum / beta

def _smooth_objective(P, r_target, beta=50.0):
    """
    Smooth surrogate for the packing objective using soft-min.
    Minimizes: -softmin(min_pair_dist/2, min_boundary_dist)
    plus penalty for violations.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    pair_dists = dist / 2.0
    boundary_dists = 1.0 - np.linalg.norm(P, axis=1)
    
    all_constraints = np.concatenate([pair_dists.ravel(), boundary_dists])
    all_constraints = all_constraints[all_constraints > -1.0]  # filter out -inf
    
    if len(all_constraints) == 0:
        return 1e10
    
    # Soft-min of all constraint values
    r_soft = _softmin_logsumexp(all_constraints, beta)
    
    # Penalty for any constraint being violated (negative)
    violations = np.maximum(-all_constraints, 0.0)
    violation_penalty = np.sum(violations**2)
    
    return -r_soft + violation_penalty

def _smooth_objective_grad(P, r_target, beta=50.0):
    """
    Gradient of the smooth objective.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    pair_dists = dist / 2.0
    boundary_dists = 1.0 - np.linalg.norm(P, axis=1)
    
    all_constraints = np.concatenate([pair_dists.ravel(), boundary_dists])
    valid_mask = all_constraints > -1.0
    all_constraints_valid = all_constraints[valid_mask]
    
    if len(all_constraints_valid) == 0:
        return np.zeros_like(P)
    
    # Soft-min gradient
    x_min = np.min(all_constraints_valid)
    shifted = all_constraints_valid - x_min
    exp_vals = np.exp(-beta * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(P)
    
    # Gradient from pair distances
    n_pairs = N * (N - 1) // 2
    # Reconstruct pair index mapping
    i_idx, j_idx = np.triu_indices(N, k=1)
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        if w < 1e-15:
            continue
        d_vec = P[i] - P[j]
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-15:
            continue
        grad[i] += w * d_vec / d_norm
        grad[j] -= w * d_vec / d_norm
    
    # Gradient from boundary distances
    for i in range(N):
        idx = n_pairs + i
        w = weights[idx]
        if w < 1e-15:
            continue
        p_norm = np.linalg.norm(P[i])
        if p_norm < 1e-15:
            continue
        grad[i] -= w * P[i] / p_norm
    
    # Violation penalty gradient
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        d = dist[i, j]
        if d < 2 * r_target:
            d_vec = P[i] - P[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-15:
                continue
            penalty_w = (2 * r_target - d)
            grad[i] -= penalty_w * d_vec / d_norm
            grad[j] += penalty_w * d_vec / d_norm
    
    for i in range(N):
        p_norm = np.linalg.norm(P[i])
        if p_norm > 1.0:
            penalty_w = (p_norm - 1.0)
            if p_norm < 1e-15:
                continue
            grad[i] += penalty_w * P[i] / p_norm
    
    return grad

def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))

def _energy_grad(P, r):
    """
    Gradient of the energy function.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    W = V_cc * inv_dist
    
    sum_W = np.sum(W, axis=1) 
    term1 = (P.T @ W).T 
    term2 = P * sum_W[:, np.newaxis] 
    
    grad += term1 - term2
    
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
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

def _optimize_positions(P_init, r_target, time_limit=5.0):
    """
    Optimize positions P to minimize energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())

def _optimize_smooth(P_init, r_target, beta_start=20.0, beta_end=200.0, time_limit=10.0):
    """
    Progressive annealing optimization using smooth soft-min objective.
    """
    start_time = time.time()
    P_curr = P_init.copy()
    n_steps = 5
    
    for step in range(n_steps):
        if time.time() > start_time + time_limit:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (step / max(n_steps - 1, 1))
        
        def objective(P_flat):
            P = P_flat.reshape(N, 2)
            return _smooth_objective(P, r_target, beta)
        
        def grad(P_flat):
            P = P_flat.reshape(N, 2)
            return _smooth_objective_grad(P, r_target, beta).flatten()
        
        try:
            res = minimize(objective, P_curr.flatten(), jac=grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            P_new = res.x.reshape(N, 2)
            score_new = _compute_score(P_new)
            score_curr = _compute_score(P_curr)
            if score_new > score_curr:
                P_curr = P_new
        except Exception:
            pass
    
    return P_curr

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

def _high_precision_polish(P_init, r_init, deadline):
    """
    High-precision final polish with very tight tolerances.
    """
    P_curr = P_init.copy()
    r_curr = r_init
    
    # Multiple L-BFGS passes with decreasing tolerance
    for tol in [1e-10, 1e-12, 1e-14]:
        if time.time() > deadline - 3.0:
            break
        
        def objective(P_flat):
            P_test = P_flat.reshape(N, 2)
            score = _compute_score(P_test)
            norms = np.linalg.norm(P_test, axis=1)
            outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2)
            return -score + outside_penalty * 1000
        
        try:
            result = minimize(objective, P_curr.flatten(), method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': tol, 'gtol': tol})
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
        except Exception:
            pass
    
    # Final coordinate descent refinement
    for it in range(200):
        if time.time() > deadline - 2.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            
            P_test = P_curr.copy()
            best_pos = P_curr[i].copy()
            best_score = r_curr
            
            # Try small perturbations
            for dx in [-0.001, -0.0005, 0.0005, 0.001]:
                for dy in [-0.001, -0.0005, 0.0005, 0.001]:
                    P_test[i] = P_curr[i] + np.array([dx, dy])
                    s = _compute_score(P_test)
                    if s > best_score:
                        best_score = s
                        best_pos = P_test[i].copy()
            
            if not np.allclose(best_pos, P_curr[i]):
                P_curr[i] = best_pos
                improved = True
        
        if not improved:
            break
    
    r_final = _compute_score(P_curr)
    if r_final > r_curr:
        r_curr = r_final
    
    return P_curr, r_curr

def _local_refine(P_init, r_init, deadline):
    """
    Local refinement using gradient-based optimization with adaptive steps.
    """
    P_curr = P_init.copy()
    r_curr = r_init
    
    # Use scipy's L-BFGS-B for more effective local optimization
    def objective(P_flat):
        P_test = P_flat.reshape(N, 2)
        score = _compute_score(P_test)
        norms = np.linalg.norm(P_test, axis=1)
        outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2)
        return -score + outside_penalty * 100
    
    P_flat = P_curr.flatten()
    
    try:
        result = minimize(objective, P_flat, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        if result.fun < -r_curr + 1e-10:
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
    except:
        pass
    
    # Additional manual refinement pass
    max_iter = 100
    tol = 1e-12
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
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
        for factor in [0.1, 0.05, 0.02, 0.01, 0.005]:
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
        
        if r_curr - best_score < tol:
            break
    
    return P_curr, r_curr

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer
    
    np.random.seed(42)
    
    # Define ring structures - more diverse
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 11, 15, 17],
    ]
    
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    for ring_counts in ring_structures:
        if time.time() > deadline - 35.0:
            break
            
        # 1. Global search for ring parameters
        P_candidate = _global_search_ring_params(ring_counts, time_limit=15.0)
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Smooth annealing optimization
        if time.time() < deadline - 20.0:
            P_smooth = _optimize_smooth(P_candidate, score_best, 
                                         beta_start=15.0, beta_end=150.0, 
                                         time_limit=12.0)
            score_smooth = _compute_score(P_smooth)
            if score_smooth > score_best:
                P_best = P_smooth
                score_best = score_smooth
        
        # 3. Local refinement with binary search on r
        r_low = score_best * 0.95
        r_high = 0.15
        
        P_opt, E = _optimize_positions(P_best, r_low, time_limit=4.0)
        if E < 1e-6:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
            r_low = score
            
        # Binary search
        for _ in range(6):
            if time.time() > deadline - 12.0:
                break
                
            r_mid = (r_low + r_high) / 2.0
            
            P_opt, E = _optimize_positions(P_best, r_mid, time_limit=4.0)
            if E < 1e-8:
                r_low = r_mid
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt
                    score_best = score
            else:
                r_high = r_mid
        
        # 4. Final local refinement
        if time.time() < deadline - 8.0:
            P_ref, score_ref = _local_refine(P_best, score_best, deadline)
            if score_ref > score_best:
                P_best = P_ref
                score_best = score_ref
    
    # 5. High-precision final polish
    if time.time() < deadline - 5.0:
        P_polished, score_polished = _high_precision_polish(P_best, score_best, deadline)
        if score_polished > score_best:
            P_best = P_polished
            score_best = score_polished
    
    # Final safety check
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
    
    final_score = _compute_score(P_best)
    if not np.isfinite(final_score) or final_score <= 0:
        P_best = _get_hex_initial_points()
    
    return P_best
# EVOLVE-BLOCK-END
