"""Improved program for packing 300 circles in a unit circle using force-directed relaxation."""
import numpy as np
import time
from scipy.spatial.distance import cdist
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
    max_k = int(1.0 / (s * 0.5)) + 3
    
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
        max_k = int(1.0 / (s * 0.5)) + 4
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

def _force_directed_relax(points, max_iters=500, time_limit=None, start_time=None):
    """
    Force-directed relaxation: push overlapping circles apart, pull boundary-violating circles inward.
    Uses adaptive step size and neighbor-based forces.
    """
    n = len(points)
    points = points.copy()
    
    if start_time is None:
        start_time = time.time()
    
    # Precompute upper triangle indices
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    
    # Initial step size
    step = 0.01
    min_step = 1e-8
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Compute all pair distances
        vecs = points[i_idx] - points[j_idx]
        dists = np.sqrt(np.sum(vecs**2, axis=1))
        
        # Compute boundary violations
        norms = np.linalg.norm(points, axis=1)
        boundary_viol = norms - 1.0  # positive if outside
        
        # Find the minimum distance (half of min pair dist) and min boundary dist
        min_pair_dist = np.min(dists) / 2.0
        min_boundary_dist = 1.0 - np.max(norms)
        current_r = min(min_pair_dist, min_boundary_dist)
        
        # If no violations, we're done
        if min_pair_dist > 0.053 and min_boundary_dist > 0.053:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pair forces: push apart if closer than 2*r_target
        # Use a target radius slightly above current to encourage growth
        r_target = current_r * 1.02 + 0.0001
        
        # Only apply forces to pairs that are too close
        close_mask = dists < 2.0 * r_target
        if np.any(close_mask):
            close_i = i_idx[close_mask]
            close_j = j_idx[close_mask]
            close_vecs = vecs[close_mask]
            close_dists = dists[close_mask]
            
            # Force magnitude proportional to overlap
            overlap = 2.0 * r_target - close_dists
            # Normalize direction
            safe_dists = np.where(close_dists > 1e-12, close_dists, 1.0)
            dirs = close_vecs / safe_dists[:, np.newaxis]
            
            # Force = direction * overlap * weight
            weights = overlap / safe_dists  # stronger for closer pairs
            force_mags = weights * 0.5
            
            # Apply forces
            np.add.at(forces, close_i, -force_mags[:, np.newaxis] * dirs)
            np.add.at(forces, close_j, force_mags[:, np.newaxis] * dirs)
        
        # Boundary forces: pull inward if outside
        outside_mask = boundary_viol > 0
        if np.any(outside_mask):
            outside_idx = np.where(outside_mask)[0]
            outside_norms = norms[outside_idx]
            outside_vecs = points[outside_idx]
            
            # Force direction is inward (toward origin)
            safe_norms = np.where(outside_norms > 1e-12, outside_norms, 1.0)
            dirs = outside_vecs / safe_norms[:, np.newaxis]
            
            # Force magnitude proportional to violation
            force_mags = boundary_viol[outside_idx] * 0.5
            forces[outside_idx] -= force_mags[:, np.newaxis] * dirs
        
        # Compute step size adaptively
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            # Scale step to not move too far
            step = min(step, 0.1 / max_force)
        
        # Update positions
        points += step * forces
        
        # Project back inside if needed
        new_norms = np.linalg.norm(points, axis=1)
        outside = new_norms > 1.0
        if np.any(outside):
            points[outside] /= new_norms[outside, np.newaxis]
        
        # Decay step size
        step *= 0.995
        if step < min_step:
            step = min_step
    
    return points

def _local_polish(points, time_limit=10.0):
    """
    Local refinement using L-BFGS-B with a smooth objective on the best configuration.
    """
    n = len(points)
    points = points.copy()
    start_time = time.time()
    
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z = max(np.max(z_pairs), np.max(z_bounds))
        
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
    
    # Try a few beta values
    betas = [500.0, 2000.0, 10000.0]
    
    best_points = points
    best_score = _compute_score(points)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-12,
                    'ftol': 1e-14,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N)]
    
    best_points = None
    best_score = -np.inf
    
    for init_points in inits:
        if time.time() > deadline - 30:
            break
        
        # Apply force-directed relaxation
        relaxed = _force_directed_relax(
            init_points, 
            max_iters=1000,
            time_limit=deadline - time.time() - 20,
            start_time=start_time
        )
        
        score = _compute_score(relaxed)
        
        if score > best_score:
            best_points = relaxed
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
    
    # Local polish with L-BFGS-B if time permits
    if time.time() < deadline - 5:
        try:
            polished = _local_polish(best_points, time_limit=deadline - time.time() - 2)
            polished_score = _compute_score(polished)
            if polished_score > best_score:
                best_points = polished
                best_score = polished_score
        except Exception:
            pass
    
    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
