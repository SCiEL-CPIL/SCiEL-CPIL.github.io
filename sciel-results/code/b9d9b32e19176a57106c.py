"""Improved program for packing 300 circles in a unit circle using Repulsion Relaxation followed by Soft-Min Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    points += np.random.normal(0, 1e-4, points.shape)
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings.
    """
    points = []
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    diff = n - sum(ring_counts)
    if diff > 0:
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        for i in range(len(ring_counts)-1, -1, -1):
            remove = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= remove
            diff += remove
            if diff >= 0:
                break
    
    while sum(ring_counts) > n:
        for i in range(len(ring_counts)-1, -1, -1):
            if ring_counts[i] > 1:
                ring_counts[i] -= 1
                break
    while sum(ring_counts) < n:
        ring_counts[-1] += 1
    
    idx = 0
    for k, count in enumerate(ring_counts):
        if k == 0:
            if count >= 1:
                points.append([0.0, 0.0])
                idx += 1
        else:
            radius = k * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
            phase = (k % 2) * np.pi / count
            for m in range(count):
                if idx >= n:
                    break
                angle = 2 * np.pi * m / count + phase
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                if x*x + y*y <= 1.0:
                    points.append([x, y])
                    idx += 1
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x*x + y*y <= 1.0:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    
    points += np.random.normal(0, 1e-4, points.shape)
    return points

def _relax_repulsion(points, max_iter=200, deadline=None):
    """
    Relax points using repulsive forces (inverse square law).
    Minimizes E = sum 1/d^2.
    """
    n = len(points)
    p = points.copy()
    
    # Precompute indices for efficiency? No, vectorize the whole thing.
    # For N=300, N^2 = 90,000. This is fine.
    
    current_energy = np.inf
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 10:
            break
            
        # Compute pairwise vectors and distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero
        
        dists = np.sqrt(dists_sq)
        
        # Pairwise forces: F_ij = 2 * diff_ij / dists^3
        # We sum over j for each i.
        # force_i = sum_j 2 * (p_i - p_j) / d_ij^3
        # This is equivalent to:
        # forces = 2 * sum_j (diff_ij / dists^3)
        
        # To avoid memory explosion for N=300, we can just compute it.
        # dists^3
        inv_dists_cubed = 1.0 / (dists_sq * dists)
        
        # Zero out diagonal
        np.fill_diagonal(inv_dists_cubed, 0.0)
        
        # Pairwise force contribution
        # F_pair = 2 * sum_j diff_ij * inv_dists_cubed_ij
        # diff is (N, N, 2). inv_dists_cubed is (N, N).
        # We need to multiply diff by inv_dists_cubed[..., np.newaxis]
        
        F_pair = 2.0 * np.sum(diff * inv_dists_cubed[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        norms = np.linalg.norm(p, axis=1)
        # F_bound = 2 * p / (norm * (1-norm)^3)
        # Careful with norms close to 0 or 1
        safe_norms = np.where(norms > 1e-6, norms, 1.0)
        safe_bound = np.where(1.0 - norms > 1e-6, 1.0 - norms, 1.0)
        
        F_bound = 2.0 * p / (safe_norms[:, np.newaxis] * safe_bound[:, np.newaxis]**3)
        
        # Total force
        F_total = F_pair + F_bound
        
        # Gradient descent step
        # Adaptive step size
        step = 1e-4
        # Simple line search: try step, if energy increases, halve it.
        # But we don't want to compute energy (sum 1/d^2) explicitly every time if we can avoid it?
        # Actually, computing energy is O(N^2). We already have dists_sq.
        
        # Calculate current energy
        inv_dists_sq = 1.0 / dists_sq
        np.fill_diagonal(inv_dists_sq, 0.0)
        E_pair = np.sum(inv_dists_sq)
        
        E_bound = np.sum(1.0 / safe_bound**2)
        
        current_energy = E_pair + E_bound
        
        # Move
        p_new = p + step * F_total
        
        # Check if points went outside?
        # The force should push them in, but let's clamp.
        norms_new = np.linalg.norm(p_new, axis=1)
        outside = norms_new > 0.99
        if np.any(outside):
            p_new[outside] = p_new[outside] / norms_new[outside, np.newaxis] * 0.99
            
        # Calculate new energy (approximate or exact?)
        # Exact is safer.
        diff_new = p_new[:, np.newaxis, :] - p_new[np.newaxis, :, :]
        dists_sq_new = np.sum(diff_new**2, axis=-1)
        np.fill_diagonal(dists_sq_new, 1.0)
        inv_dists_sq_new = 1.0 / dists_sq_new
        np.fill_diagonal(inv_dists_sq_new, 0.0)
        E_pair_new = np.sum(inv_dists_sq_new)
        
        norms_new = np.linalg.norm(p_new, axis=1)
        safe_bound_new = np.where(1.0 - norms_new > 1e-6, 1.0 - norms_new, 1.0)
        E_bound_new = np.sum(1.0 / safe_bound_new**2)
        
        new_energy = E_pair_new + E_bound_new
        
        if new_energy < current_energy:
            p = p_new
            current_energy = new_energy
        else:
            # Reduce step size
            step *= 0.5
            if step < 1e-10:
                break
                
    return p

def _soft_min_objective_and_grad(x, beta, iu):
    p = x.reshape(-1, 2)
    i_idx, j_idx = iu
    vecs = p[i_idx] - p[j_idx]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
    
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    z_pairs = -beta * pair_dists
    z_bounds = -beta * bound_dists
    
    max_z_pair = np.max(z_pairs)
    max_z_bound = np.max(z_bounds)
    max_z = max(max_z_pair, max_z_bound)
    
    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)
    
    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    
    log_sum_exp = max_z + np.log(sum_exp)
    
    val = log_sum_exp / beta
    
    w_pairs = exp_pairs / sum_exp
    w_bounds = exp_bounds / sum_exp
    
    grad = np.zeros_like(p)
    
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    coeff = w_pairs / (2.0 * safe_pair_dists)
    
    np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
    
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = p / safe_norms[:, np.newaxis]
    
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    iu = np.triu_indices(N, k=1)
    
    best_points = None
    best_score = -np.inf
    
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
    for init_points in inits:
        if time.time() > deadline - 30:
            break
            
        # Phase 1: Repulsion Relaxation
        relaxed_points = _relax_repulsion(init_points, max_iter=100, deadline=deadline)
        
        # Phase 2: Soft-Min Polish
        current_points = relaxed_points
        score = _compute_score(current_points)
        
        # Run L-BFGS-B with increasing beta
        betas = [50.0, 100.0, 500.0, 2000.0, 10000.0]
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    lambda x: _soft_min_objective_and_grad(x, beta, iu)[0],
                    current_points.flatten(),
                    jac=lambda x: _soft_min_objective_and_grad(x, beta, iu)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14, 'disp': False}
                )
                
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > score:
                    current_points = candidate
                    score = candidate_score
            except Exception:
                pass
        
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
            
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
        
    # Final safety check
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        
    return best_points

# EVOLVE-BLOCK-END
