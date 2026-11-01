"""Optimized packing of 50 equal circles in a unit circle using soft-min L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 5.0  # seconds before the 150s kill to stop

def _compute_score(pts):
    """Compute the packing radius for given centers."""
    n = len(pts)
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    # Use vectorized computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _soft_min_obj_and_grad(pts, beta):
    """
    Objective: minimize -soft_min(constraints) where constraints are
    [pairwise distances / 2, distance to boundary]
    We maximize the minimum of all constraints.
    
    soft_min(x, beta) = -1/beta * log(sum(exp(-beta * x_i)))
    We want to maximize soft_min, so minimize -soft_min.
    
    For numerical stability, we shift by the min.
    """
    n = len(pts)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    
    # Boundary constraints
    boundary_constraints = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Combine all constraints
    # We'll work with the flat array of all constraints
    pair_flat = pair_constraints[np.triu_indices(n, k=1)]
    all_constraints = np.concatenate([pair_flat, boundary_constraints])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x_i)))
    # For numerical stability, subtract the min first
    min_c = np.min(all_constraints)
    shifted = all_constraints - min_c
    # exp(-beta * shifted) where shifted >= 0
    # If beta is large, exp(-beta * shifted) underflows for shifted > ~500/beta
    # Use log-sum-exp trick
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = -min_c - log_sum / beta
    
    # Gradient
    # d(soft_min)/d(pts) = sum_i (exp(-beta*(c_i - min_c)) / sum_j exp(-beta*(c_j - min_c))) * (-beta) * d(c_i)/d(pts)
    # = -sum_i (w_i) * d(c_i)/d(pts) where w_i = exp(-beta*(c_i - min_c)) / sum
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of pairwise constraints
    # c_ij = |p_i - p_j| / 2
    # d(c_ij)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|) = (p_i - p_j) / (2 * dist_ij)
    # d(c_ij)/d(p_j) = (p_j - p_i) / (2 * |p_i - p_j|) = -(p_i - p_j) / (2 * dist_ij)
    
    grad = np.zeros_like(pts)
    
    # For each pair (i, j) with i < j:
    # weight w_ij
    # d(c_ij)/d(p_i) = (p_i - p_j) / (2 * dist_ij)
    # d(c_ij)/d(p_j) = (p_j - p_i) / (2 * dist_ij)
    
    # Vectorized approach
    # For pair (i,j): contribution to grad[i] += w_ij * (p_i - p_j) / (2 * dist_ij)
    #                contribution to grad[j] += w_ij * (p_j - p_i) / (2 * dist_ij)
    
    # We need to accumulate gradients
    # diff[i,j] = p_i - p_j
    # dists[i,j] = |p_i - p_j|
    
    # For i < j:
    # grad[i] += w_ij * (p_i - p_j) / (2 * dists[i,j])
    # grad[j] += w_ij * (p_j - p_i) / (2 * dists[i,j]) = -w_ij * (p_i - p_j) / (2 * dists[i,j])
    
    # Let's compute the contribution matrix
    # contrib[i,j] = w_ij * (p_i - p_j) / (2 * dists[i,j]) for i != j
    # But we only use upper triangle for pairs
    
    # Actually, let's be more careful. The soft_min is over all constraints.
    # The gradient of soft_min w.r.t. pts is:
    # sum_k w_k * (-beta) * d(c_k)/d(pts)
    # where w_k = exp(-beta*(c_k - min_c)) / sum_j exp(-beta*(c_j - min_c))
    
    # Wait, let me re-derive.
    # soft_min = -1/beta * log(sum_k exp(-beta * c_k))
    # d(soft_min)/d(c_k) = -1/beta * (-beta) * exp(-beta*c_k) / sum_j exp(-beta*c_j)
    # = exp(-beta*c_k) / sum_j exp(-beta*c_j)
    # = w_k (normalized weights)
    
    # So d(soft_min)/d(pts) = sum_k w_k * d(c_k)/d(pts)
    
    # For pair constraint c_ij = |p_i - p_j| / 2:
    # d(c_ij)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
    # d(c_ij)/d(p_j) = (p_j - p_i) / (2 * |p_i - p_j|)
    
    # For boundary constraint c_i = 1 - |p_i|:
    # d(c_i)/d(p_i) = -p_i / |p_i|
    
    # So the total gradient is:
    # sum_{i<j} w_ij * [(p_i - p_j) / (2 * dist_ij) for p_i, (p_j - p_i) / (2 * dist_ij) for p_j]
    # + sum_i w_boundary_i * (-p_i / |p_i|)
    
    # We want to MINIMIZE -soft_min, so the gradient of the objective is -d(soft_min)/d(pts)
    
    # Let's compute d(soft_min)/d(pts) first, then negate.
    
    grad_soft_min = np.zeros_like(pts)
    
    # Pair contributions
    # For i < j, the weight is pair_weights[i,j] (from the upper triangle of weights)
    # We need to map the flat weights back to pairs
    
    # Let's re-index. pair_flat[k] corresponds to pair (i,j) where k is the index in triu
    # We need to know which (i,j) each flat index corresponds to
    
    # Alternative: compute the gradient contribution directly
    # For each pair (i,j) with i<j:
    #   weight = weights_pair[i,j]
    #   dir_ij = (p_i - p_j) / dists[i,j]  (unit vector from j to i)
    #   grad_soft_min[i] += weight * dir_ij / 2
    #   grad_soft_min[j] -= weight * dir_ij / 2
    
    # For boundary:
    #   weight = weights_boundary[i]
    #   grad_soft_min[i] += weight * (-p_i / |p_i|)
    
    # Let's compute pair weights matrix
    pair_weights = np.zeros((n, n))
    # Fill upper triangle
    iu, ju = np.triu_indices(n, k=1)
    pair_weights[iu, ju] = pair_flat_weights = weights[:len(pair_flat)]
    pair_weights[ju, iu] = pair_flat_weights
    
    # Boundary weights
    boundary_weights = weights[len(pair_flat):]
    
    # Now compute gradient
    # For pairs:
    # unit[i,j] = (p_i - p_j) / dists[i,j]
    # Handle dists[i,j] = 0 (shouldn't happen if points are distinct)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    unit = diff / safe_dists[:, :, np.newaxis]
    
    # grad_soft_min[i] = sum_j pair_weights[i,j] * unit[i,j] / 2
    # This is a matrix-vector product
    grad_soft_min += np.sum(pair_weights[:, :, np.newaxis] * unit, axis=1) / 2.0
    
    # For boundary:
    # grad_soft_min[i] += boundary_weights[i] * (-p_i / |p_i|)
    norm_pts = np.linalg.norm(pts, axis=1, keepdims=True)
    safe_norm = np.where(norm_pts > 1e-10, norm_pts, 1.0)
    grad_soft_min += boundary_weights[:, np.newaxis] * (-pts / safe_norm)
    
    # Objective is -soft_min, so gradient is -grad_soft_min
    obj_val = -soft_min_val
    grad_val = -grad_soft_min
    
    return obj_val, grad_val

def _ring_init(n_circles=50):
    """Initialize points in concentric rings."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    
    # Ring 1: 6 points
    r1 = 0.15
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 points
    r2 = 0.30
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12  # offset
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 points
    r3 = 0.45
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 points (to reach 50)
    r4 = 0.58
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        pts.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    pts = np.array(pts[:n_circles])
    # Add small random perturbation to break symmetry
    pts += np.random.randn(*pts.shape) * 0.01
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial points
    pts = _ring_init(N)
    
    # Phase 1: Coarse optimization with low beta
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    beta_values = [50, 200, 1000, 5000, 10000]
    
    for beta in beta_values:
        if time.time() > deadline:
            break
        
        # Optimize with current beta
        result = minimize(
            _soft_min_obj_and_grad,
            pts,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.status == 1:
            new_pts = result.x
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If optimization failed, keep current pts
            pass
    
    # Phase 2: Fine polish with high beta
    if time.time() < deadline:
        for beta in [10000, 20000, 50000]:
            if time.time() > deadline:
                break
            
            result = minimize(
                _soft_min_obj_and_grad,
                best_pts,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success or result.status == 1:
                new_pts = result.x
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
    
    return best_pts
