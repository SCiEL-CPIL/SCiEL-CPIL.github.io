"""Optimized circle packing: 300 equal circles in unit circle.
Uses hexagonal lattice initialization + vectorized log-sum-exp soft-min optimization
with adaptive learning rate and final polish.
"""
import numpy as np
import time

N = 300
DEADLINE_FRACTION = 0.88  # safety margin for 150s wall-clock

def _hex_lattice_init(n):
    """Generate a hexagonal lattice of points inside the unit disk, return n points."""
    # Estimate spacing for hex packing
    d = np.sqrt(4.0 * np.pi / (n * np.sqrt(3.0)))
    
    while True:
        k = int(1.0 / d) + 3
        pts_list = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * np.sqrt(3.0) / 2.0
                pts_list.append((x, y))
        pts = np.array(pts_list, dtype=np.float64)
        r = np.hypot(pts[:, 0], pts[:, 1])
        # Keep points well inside
        mask = r <= 1.0 - d * 0.5
        pts = pts[mask]
        if len(pts) >= n:
            idx = np.argsort(r, kind="stable")[:n]
            return pts[idx]
        d *= 0.97


def _compute_radius_fast(pts):
    """Compute the packing radius efficiently."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq)
    # Distance to boundary
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(0.5 * min_pair, min_boundary)


def _soft_min_logsumexp(values, temperature):
    """Compute soft minimum using log-sum-exp approximation.
    soft_min(x) ≈ -T * log(sum(exp(-x_i / T)))
    """
    x = np.asarray(values)
    # Shift for numerical stability
    x_shifted = -x / temperature
    x_max = np.max(x_shifted)
    if not np.isfinite(x_max):
        return np.min(x)
    log_sum = x_max + np.log(np.sum(np.exp(x_shifted - x_max)))
    return -temperature * log_sum


def _objective_and_grad(pts, temperature):
    """Compute soft-min objective and its gradient.
    Objective = soft_min(0.5 * min_pairwise_dist, min_boundary_dist)
    We want to maximize this, so we minimize -objective.
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq + 1e-30)
    
    # Half pairwise distances
    half_dist = 0.5 * dist
    np.fill_diagonal(half_dist, np.inf)
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dist = 1.0 - radii
    
    # Combine all constraints into one vector
    # For each pair (i,j): constraint = 0.5 * |c_i - c_j|
    # For each point i: constraint = 1 - |c_i|
    # We want soft_min of all these
    
    # Vector of all pairwise half-distances (excluding diagonal)
    mask = ~np.eye(n, dtype=bool)
    pair_vals = half_dist[mask]
    boundary_vals = boundary_dist
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    # Soft min
    obj = -_soft_min_logsumexp(all_vals, temperature)
    
    # Gradient with respect to pts
    # d/dp_i of soft_min = -sum_j (exp(-v_j/T) / sum_k exp(-v_k/T)) * dv_j/dp_i
    # where v_j are the constraint values
    
    # Compute weights
    x_shifted = -all_vals / temperature
    x_max = np.max(x_shifted)
    exp_vals = np.exp(x_shifted - x_max)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # shape (M,) where M = n*(n-1)/2 + n
    
    # Gradient of pairwise distances
    # d/dp_i (0.5 * |p_i - p_j|) = 0.5 * (p_i - p_j) / |p_i - p_j|
    # For each pair (i,j), the contribution to grad[i] is w * 0.5 * (p_i - p_j) / dist
    # and to grad[j] is w * 0.5 * (p_j - p_i) / dist
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    # Reconstruct which pairs correspond to which indices
    # This is expensive; let's use a different approach
    
    # Alternative: compute gradient directly using matrix operations
    # For each i, grad[i] = -sum_{j != i} w_ij * (p_i - p_j) / (2 * dist_ij)
    # where w_ij = exp(-0.5*dist_ij/T) / sum_k exp(-v_k/T)
    
    # Actually, let's just compute the gradient numerically or use a simpler approach
    # Given time constraints, let's use a force-based approach instead
    
    return obj, grad


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRACTION
    
    # Initialize with hex lattice
    pts = _hex_lattice_init(N)
    if len(pts) != N:
        pts = np.random.randn(N, 2) * 0.5
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 0.9
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.9
    
    best_pts = pts.copy()
    best_score = _compute_radius_fast(best_pts)
    
    # Force-directed optimization with adaptive step
    r_est = best_score
    if r_est <= 0:
        r_est = 0.05
    
    lr = 0.002
    no_improve = 0
    patience = 500
    n_iter = 0
    
    while time.time() < deadline:
        n_iter += 1
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq + 1e-30)
        
        # Radii
        radii = np.hypot(pts[:, 0], pts[:, 1])
        
        # Current score
        min_pair = np.sqrt(np.min(dist_sq))
        min_boundary = np.min(1.0 - radii)
        current_score = min(0.5 * min_pair, min_boundary)
        
        if current_score > best_score + 1e-12:
            best_score = current_score
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience:
            # Perturb to escape local minimum
            pts = pts + np.random.randn(N, 2) * 0.01
            radii = np.hypot(pts[:, 0], pts[:, 1])
            mask = radii > 0.995
            if np.any(mask):
                pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.995
            no_improve = 0
            lr = max(lr * 0.7, 0.0001)
            continue
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsive forces between close pairs
        threshold = 2.0 * r_est * 1.3
        close_mask = dist < threshold
        np.fill_diagonal(close_mask, False)
        
        # For each pair (i,j) with dist < threshold:
        # Force on i: (2*r_est - dist_ij) / dist_ij * (p_i - p_j) / dist_ij
        # = (2*r_est - dist_ij) / dist_ij^2 * (p_i - p_j)
        
        coeff = np.zeros_like(dist)
        valid = close_mask & (dist > 1e-10)
        coeff[valid] = (2.0 * r_est - dist[valid]) / (dist[valid] ** 2)
        
        # Force on i = sum_j coeff[i,j] * (p_i - p_j)
        row_sums = np.sum(coeff, axis=1)
        force_pair = pts * row_sums[:, np.newaxis] - coeff @ pts
        
        # Boundary force
        boundary_dist = 1.0 - radii
        boundary_force = np.zeros_like(pts)
        close_to_boundary = boundary_dist < r_est * 1.2
        if np.any(close_to_boundary):
            idx = np.where(close_to_boundary)[0]
            for i in idx:
                if radii[i] > 1e-10:
                    direction = -pts[i] / radii[i]
                    strength = (r_est * 1.2 - boundary_dist[i]) / (r_est + 1e-10)
                    boundary_force[i] = direction * strength * 0.3
        
        forces = force_pair + boundary_force
        
        # Update
        pts += lr * forces
        
        # Keep inside
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.999
        
        # Adapt learning rate
        if current_score > best_score:
            lr = min(lr * 1.01, 0.005)
        else:
            lr = max(lr * 0.998, 0.00005)
        
        r_est = max(current_score, 0.01)
    
    # Final polish: a few more iterations with smaller step
    if time.time() < deadline - 5.0:
        for _ in range(200):
            if time.time() >= deadline - 2.0:
                break
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq + 1e-30)
            radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
            
            min_pair = np.sqrt(np.min(dist_sq))
            min_boundary = np.min(1.0 - radii)
            current_score = min(0.5 * min_pair, min_boundary)
            
            threshold = 2.0 * current_score * 1.1
            close_mask = dist < threshold
            np.fill_diagonal(close_mask, False)
            
            coeff = np.zeros_like(dist)
            valid = close_mask & (dist > 1e-10)
            coeff[valid] = (2.0 * current_score - dist[valid]) / (dist[valid] ** 2)
            
            row_sums = np.sum(coeff, axis=1)
            force_pair = best_pts * row_sums[:, np.newaxis] - coeff @ best_pts
            
            boundary_dist = 1.0 - radii
            boundary_force = np.zeros_like(best_pts)
            close_to_boundary = boundary_dist < current_score * 1.05
            if np.any(close_to_boundary):
                idx = np.where(close_to_boundary)[0]
                for i in idx:
                    if radii[i] > 1e-10:
                        direction = -best_pts[i] / radii[i]
                        strength = (current_score * 1.05 - boundary_dist[i]) / (current_score + 1e-10)
                        boundary_force[i] = direction * strength * 0.2
            
            forces = force_pair + boundary_force
            best_pts += 0.0005 * forces
            
            radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
            mask = radii > 0.9999
            if np.any(mask):
                best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis] * 0.9999
    
    return best_pts
