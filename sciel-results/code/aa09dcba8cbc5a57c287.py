import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init():
    """Generate initial hexagonal lattice points inside unit circle."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[radius <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            radius = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[np.argsort(radius, kind="stable")[:N]]
            return pts
        spacing *= 0.98

def _compute_score(pts):
    """Compute the packing radius (minimum of pairwise half-distances and boundary distances)."""
    if len(pts) < 2:
        return min(1.0 - np.linalg.norm(pts, axis=1)) if len(pts) > 0 else 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
    return min(min_pair, min_bound)

def _soft_min_objective_and_grad(x, tau=50.0):
    """
    Compute soft-min objective and gradient.
    We want to maximize the minimum of all pairwise distances and boundary distances.
    softmin(S) = -1/tau * log(sum(exp(-tau * s)))
    We minimize -softmin(S).
    """
    pts = x.reshape(-1, 2)
    N_pts = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    rad = np.linalg.norm(pts, axis=1)
    bound = 1.0 - rad
    
    # Collect all distances
    all_dists = np.concatenate([dist[dist < np.inf].flatten(), bound])
    
    if len(all_dists) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft-min value
    # f = -1/tau * log(sum(exp(-tau * s)))
    # For numerical stability: m = max(-tau * s)
    neg_tau_s = -tau * all_dists
    m = np.max(neg_tau_s)
    log_sum_exp = m + np.log(np.sum(np.exp(neg_tau_s - m)))
    softmin_val = -1.0 / tau * log_sum_exp
    
    # Objective is -softmin
    obj = -softmin_val
    
    # Gradient
    # d(softmin)/dx = 1/tau * sum(exp(-tau*s) * ds/dx) / sum(exp(-tau*s))
    # But we're minimizing -softmin, so grad = -d(softmin)/dx
    # = -1/tau * sum(exp(-tau*s) * ds/dx) / sum(exp(-tau*s))
    
    # Weights for each distance
    exp_terms = np.exp(neg_tau_s - m)  # exp(-tau*s) / exp(-tau*m)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # Normalized weights
    
    # Split weights back into pairwise and boundary
    n_pairs = N_pts * (N_pts - 1) // 2
    pair_weights = weights[:n_pairs]
    bound_weights = weights[n_pairs:]
    
    # Reshape pair weights to N x N (upper triangle only, but we need full for computation)
    # Actually, let's just compute the gradient directly
    # grad_pair[i] = sum_j w_ij * (x_i - x_j) / d_ij * (1/tau) ... wait
    # Let me redo this more carefully.
    
    # softmin = -1/tau * log(sum_k exp(-tau * s_k))
    # d(softmin)/dx_i = -1/tau * (1/sum_k exp(-tau*s_k)) * sum_k (-tau * exp(-tau*s_k) * ds_k/dx_i)
    # = (1/sum_k exp(-tau*s_k)) * sum_k exp(-tau*s_k) * ds_k/dx_i
    # = sum_k p_k * ds_k/dx_i where p_k = exp(-tau*s_k) / sum_j exp(-tau*s_j)
    
    # So grad(softmin) = sum_k p_k * ds_k/dx
    # And we minimize -softmin, so grad(-softmin) = -sum_k p_k * ds_k/dx
    
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    # For pair (i,j), s_ij = ||x_i - x_j||
    # ds_ij/dx_i = (x_i - x_j) / ||x_i - x_j||
    # ds_ij/dx_j = (x_j - x_i) / ||x_i - x_j||
    
    # We have pair_weights as a flat array. Let's reconstruct which pairs they correspond to.
    # This is getting complex. Let me use a simpler approach.
    
    # Alternative: compute gradient numerically? No, let's do it analytically but more carefully.
    
    # Let me just use the formula:
    # grad_i = sum_j p_ij * (x_i - x_j) / d_ij + p_bound_i * (-x_i / ||x_i||)
    # where p_ij are the normalized exp weights for pairwise distances
    
    # First, compute exp weights for all pairwise distances
    D = dist.copy()
    np.fill_diagonal(D, 0)  # Set diagonal to 0 for now
    
    # exp(-tau * D) for off-diagonal
    E_pair = np.zeros_like(D)
    mask = D > 1e-10
    E_pair[mask] = np.exp(-tau * D[mask])
    np.fill_diagonal(E_pair, 0)
    
    sum_E_pair = np.sum(E_pair)
    
    # Boundary exp weights
    E_bound = np.exp(-tau * bound)
    
    # Total sum
    total_sum = sum_E_pair + np.sum(E_bound)
    
    if total_sum < 1e-30:
        return obj, np.zeros_like(x)
    
    # Normalized weights
    p_pair = E_pair / total_sum  # N x N
    p_bound = E_bound / total_sum  # N
    
    # Pairwise gradient: grad_i += sum_j p_pair[i,j] * (x_i - x_j) / d_ij
    D_safe = np.where(D > 1e-8, D, 1.0)
    factor = p_pair / D_safe
    np.fill_diagonal(factor, 0)
    
    sum_factor = np.sum(factor, axis=1)
    weighted_sum_x = factor @ pts
    
    grad_pair = pts * sum_factor[:, np.newaxis] - weighted_sum_x
    
    # Boundary gradient: grad_i += p_bound[i] * (-x_i / ||x_i||)
    rad_safe = np.where(rad > 1e-8, rad, 1.0)
    grad_bound = -p_bound[:, np.newaxis] * pts / rad_safe[:, np.newaxis]
    
    total_grad = grad_pair + grad_bound
    
    # We minimize -softmin, so return -total_grad
    return obj, -total_grad.flatten()

def _local_polish(pts, max_iters=50, deadline=None):
    """Local polish by moving points to increase the minimum distance."""
    current_pts = pts.copy()
    current_score = _compute_score(current_pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Find the point with the smallest margin
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        rad = np.linalg.norm(current_pts, axis=1)
        bound = 1.0 - rad
        
        # For each point, compute its minimum constraint
        point_min = np.min(np.minimum(dist / 2.0, bound[:, np.newaxis]), axis=1)
        
        # Sort points by their minimum constraint
        order = np.argsort(point_min)
        
        # Try to improve the worst points first
        for idx in order[:10]:
            if deadline is not None and time.time() > deadline:
                break
            
            i = idx
            # Compute the direction to move point i
            # Move away from its nearest neighbors and towards center if near boundary
            
            # Find nearest neighbors
            row_dist = dist[i].copy()
            np.fill_diagonal(row_dist, np.inf)
            nearest = np.argmin(row_dist)
            
            # Direction away from nearest neighbor
            dir_vec = current_pts[i] - current_pts[nearest]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > 1e-10:
                dir_vec = dir_vec / dir_norm
            
            # If near boundary, add component towards center
            if bound[i] < 0.01:
                center_dir = -current_pts[i]
                center_norm = np.linalg.norm(center_dir)
                if center_norm > 1e-10:
                    center_dir = center_dir / center_norm
                    dir_vec = 0.5 * dir_vec + 0.5 * center_dir
            
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-10:
                continue
            dir_vec = dir_vec / dir_norm
            
            # Try different step sizes
            for step in [0.005, 0.002, 0.001]:
                new_pts = current_pts.copy()
                new_pts[i] += step * dir_vec
                
                # Ensure inside unit circle
                new_rad = np.linalg.norm(new_pts[i])
                if new_rad > 1.0:
                    new_pts[i] *= (1.0 - 1e-8) / new_rad
                
                new_score = _compute_score(new_pts)
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialize with hexagonal lattice
    points = _hex_lattice_init()
    current_pts = points.copy()
    current_score = _compute_score(current_pts)
    
    # 2. Optimize using soft-min surrogate with L-BFGS-B
    x0 = current_pts.flatten()
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    try:
        res = minimize(
            lambda x: _soft_min_objective_and_grad(x, tau=50.0)[0],
            x0,
            jac=lambda x: _soft_min_objective_and_grad(x, tau=50.0)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
        )
        
        if res.x is not None and time.time() < deadline - 20:
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                
    except Exception:
        pass
    
    # 3. Local polish phase
    if time.time() < deadline - 5:
        current_pts = _local_polish(current_pts, max_iters=30, deadline=deadline - 5)
    
    # Final validation
    final_score = _compute_score(current_pts)
    if final_score > 0:
        return current_pts
    else:
        return points

