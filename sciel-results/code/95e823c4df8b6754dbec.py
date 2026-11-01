"""Optimize 50 circle centres in the unit disk via soft-min + L-BFGS-B with high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin before 150s limit


def _score(points):
    """Compute the packing radius (higher is better)."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check if points are inside the unit circle
    dists_to_origin = np.sqrt(np.sum(points ** 2, axis=1))
    if np.any(dists_to_origin >= 1.0):
        return 0.0
        
    boundary_margin = 1.0 - np.max(dists_to_origin)
    if boundary_margin <= 0:
        return 0.0
        
    # Pairwise distances
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
        
    return min(min_pair / 2.0, boundary_margin)


def _soft_min_obj_and_grad(x, beta):
    """
    Soft-min objective using log-sum-exp for smoothness.
    Minimizes -r where r is the packing radius, using a smooth approximation.
    Uses vectorized gradient computation for speed.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Distances to boundary (1 - |p_i|)
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    # Combine all constraints into a single vector of "margins"
    # Margin for pair (i,j): dists[i,j] / 2
    # Margin for boundary: boundary_dists[i]
    iu = np.triu_indices(N, k=1)
    pair_margins = dists[iu] / 2.0
    
    # All margins
    all_margins = np.concatenate([pair_margins, boundary_dists])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta * m))) / beta
    # This approximates min(m) as beta -> inf
    # Numerical stability: shift by min value
    m_min = np.min(all_margins)
    shifted = all_margins - m_min
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    # Objective value
    soft_min_val = m_min - (1.0 / beta) * np.log(S)
    obj = -soft_min_val
    
    # Gradient
    weights = exp_terms / S
    
    grad = np.zeros_like(x)
    
    # Pair gradients (vectorized)
    pair_weights = weights[:len(pair_margins)]
    # Only process pairs with significant weight to save time if needed, 
    # but for N=50, 1225 pairs is small enough to always process.
    i_idx, j_idx = iu
    w_m = pair_weights
    d_m = dists[i_idx, j_idx]
    
    # Avoid division by zero
    d_safe = np.where(d_m > 1e-12, d_m, 1e-12)
    
    # Vector for each pair: (p_i - p_j) / |p_i - p_j|
    # diff[i,j] is p_i - p_j
    diff_pairs = diff[i_idx, j_idx]
    unit_vecs = diff_pairs / d_safe[:, None]
    
    # Contribution to gradient for p_i: +w * unit_vec / 2 ? 
    # Let's re-derive carefully.
    # m_ij = |p_i - p_j| / 2
    # d(m_ij)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
    # d(m_ij)/dp_j = (p_j - p_i) / (2 * |p_i - p_j|)
    
    # We are minimizing -soft_min.
    # d(-soft_min)/dp = - d(soft_min)/dp
    # d(soft_min)/dp_i = sum_k w_k * d(m_k)/dp_i
    # So d(obj)/dp_i = - sum_k w_k * d(m_k)/dp_i
    
    # For pair (i,j):
    # d(m_ij)/dp_i = 0.5 * (p_i - p_j) / d_ij
    # d(m_ij)/dp_j = -0.5 * (p_i - p_j) / d_ij
    
    # So contribution to grad[p_i] += -w_ij * 0.5 * unit_vec
    # And contribution to grad[p_j] += -w_ij * (-0.5 * unit_vec) = +w_ij * 0.5 * unit_vec
    
    contrib = w_m[:, None] * 0.5 * unit_vecs
    
    # Scatter add to i indices (negative)
    np.add.at(grad, i_idx, -contrib)
    # Scatter add to j indices (positive)
    np.add.at(grad, j_idx, contrib)
    
    # Boundary gradients
    bound_weights = weights[len(pair_margins):]
    # m_i = 1 - |p_i|
    # d(m_i)/dp_i = - p_i / |p_i|
    # d(obj)/dp_i = - w_i * d(m_i)/dp_i = w_i * p_i / |p_i|
    
    r_safe = np.where(radii > 1e-12, radii, 1e-12)
    unit_r = pts / r_safe[:, None]
    contrib_b = bound_weights[:, None] * unit_r
    
    # Scatter add to all indices
    grad += contrib_b
    
    return obj, grad


def _get_ring_initialization(n):
    """
    Generate a ring-based initialization.
    Approximate optimal distribution for N=50:
    Center: 1
    Ring 1: 6
    Ring 2: 12
    Ring 3: 12
    Ring 4: 9
    Ring 5: 10
    Total: 50
    """
    centers = []
    centers.append([0.0, 0.0])
    
    # Adjusted radii and counts to fit better
    # The radii should be roughly proportional to sqrt(count) for uniform density,
    # but packing is denser near the center.
    rings = [
        (6, 0.20),
        (12, 0.40),
        (12, 0.60),
        (9, 0.78),
        (10, 0.90)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        # Add a small offset to break symmetry if needed, but standard is fine
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
            
    return np.array(centers[:n])

def _force_directed_relax(centers, time_budget=3.0):
    """
    Pre-processing step: use force-directed relaxation to resolve overlaps
    and push points to a valid configuration before L-BFGS.
    """
    centers = centers.copy()
    start_time = time.time()
    step = 0.005
    best_centers = centers.copy()
    best_score = _score(centers)
    n = len(centers)
    
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Force magnitude: 1/(d^2 + eps)
        force_mag = 1.0 / (dists * dists + 1e-6)
        np.fill_diagonal(force_mag, 0.0)
        
        # Force vector
        force_vec = diff * force_mag[:, :, np.newaxis] / np.maximum(dists, 1e-10)[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Boundary constraint (vectorized)
        radii = np.sqrt(np.sum(centers * centers, axis=1))
        boundary_mask = radii > 0.95
        if np.any(boundary_mask):
            r_safe = np.maximum(radii, 1e-10)
            forces[boundary_mask] -= 1.0 * centers[boundary_mask] / r_safe[boundary_mask][:, np.newaxis]
        
        new_centers = centers + step * forces
        new_radii = np.sqrt(np.sum(new_centers * new_centers, axis=1))
        mask = new_radii > 1.0
        if np.any(mask):
            new_centers[mask] = new_centers[mask] / new_radii[mask][:, None] * 0.999
        
        new_score = _score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers
            step *= 1.05
        else:
            step *= 0.9
            centers = best_centers.copy()
            
        if step < 1e-8:
            break
            
    return best_centers


def _optimize_for_radius(x0, beta, maxiter=2000, gtol=1e-8, deadline=None):
    """Run L-BFGS-B optimization for a given beta."""
    def obj_and_grad(x):
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'ftol': 1e-12,
                'maxfun': maxiter * 2
            }
        )
        return result.x
    except Exception:
        return x0


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    start_time = time.time()
    
    # 1. Initialization
    pts = _get_ring_initialization(N)
    
    # 2. Force-directed relaxation to get a valid, non-overlapping start
    pts = _force_directed_relax(pts, time_budget=3.0)
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # 3. Gradient Optimization Phases
    # Phase 1: Moderate beta for global-ish search
    betas_phase1 = [50, 100, 200]
    for beta in betas_phase1:
        if time.time() - start_time > DEADLINE - 40:
            break
        x_opt = _optimize_for_radius(best_pts.flatten(), beta, maxiter=1000, gtol=1e-6)
        x_opt = x_opt.reshape(N, 2)
        # Clamp to disk
        r = np.sqrt(np.sum(x_opt ** 2, axis=1))
        mask = r > 0.99
        if np.any(mask):
            x_opt[mask] *= (0.99 / r[mask])[:, None]
        s = _score(x_opt)
        if s > best_score:
            best_score = s
            best_pts = x_opt.copy()
            
    # Phase 2: Local refinement with higher beta
    betas_phase2 = [500, 1000, 2000]
    for beta in betas_phase2:
        if time.time() - start_time > DEADLINE - 15:
            break
        x_opt = _optimize_for_radius(best_pts.flatten(), beta, maxiter=500, gtol=1e-8)
        x_opt = x_opt.reshape(N, 2)
        r = np.sqrt(np.sum(x_opt ** 2, axis=1))
        mask = r > 0.995
        if np.any(mask):
            x_opt[mask] *= (0.995 / r[mask])[:, None]
        s = _score(x_opt)
        if s > best_score:
            best_score = s
            best_pts = x_opt.copy()
            
    # Phase 3: High-precision final polish
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        betas_phase3 = [5000, 10000]
        for beta in betas_phase3:
            if time.time() - start_time > DEADLINE - 2:
                break
            x_opt = _optimize_for_radius(best_pts.flatten(), beta, maxiter=300, gtol=1e-12)
            x_opt = x_opt.reshape(N, 2)
            r = np.sqrt(np.sum(x_opt ** 2, axis=1))
            mask = r > 0.999
            if np.any(mask):
                x_opt[mask] *= (0.999 / r[mask])[:, None]
            s = _score(x_opt)
            if s > best_score:
                best_score = s
                best_pts = x_opt.copy()
                
    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback
        best_pts = _get_ring_initialization(N)
        best_pts = _force_directed_relax(best_pts, time_budget=1.0)
        
    return best_pts
