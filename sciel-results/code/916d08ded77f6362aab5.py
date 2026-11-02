"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization."""
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

def _force_based_refine(points, max_iter=5000, tol=1e-10, deadline=None):
    """
    Force-based refinement: treat circles as charged particles with repulsion
    and attraction to maintain minimum separation. This handles hard constraints
    more directly than soft-min.
    """
    points = points.copy()
    n = len(points)
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pair repulsion: push apart if too close
        vecs = points[i_idx] - points[j_idx]
        dists = np.sqrt(np.sum(vecs**2, axis=1))
        
        # Target minimum distance (use current best estimate + margin)
        target_d = 2.0 * _compute_score(points) * 1.01
        
        # Only apply force when below target
        below = dists < target_d
        if np.any(below):
            safe_d = np.where(dists[below] > 1e-12, dists[below], 1.0)
            # Force proportional to (target - dist) / dist
            force_mag = (target_d - dists[below]) / safe_d
            force_vecs = force_mag[:, np.newaxis] * vecs[below] / safe_d[:, np.newaxis]
            
            # Accumulate
            np.add.at(forces, i_idx[below], -force_vecs)
            np.add.at(forces, j_idx[below], force_vecs)
        
        # Boundary constraint: push inward if too close to boundary
        norms = np.linalg.norm(points, axis=1)
        boundary_dist = 1.0 - norms
        target_boundary = _compute_score(points) * 1.01
        below_boundary = boundary_dist < target_boundary
        if np.any(below_boundary):
            safe_norms = np.where(norms[below_boundary] > 1e-12, norms[below_boundary], 1.0)
            force_mag = (target_boundary - boundary_dist[below_boundary]) / safe_norms
            force_vecs = force_mag[:, np.newaxis] * points[below_boundary] / safe_norms[:, np.newaxis]
            forces[below_boundary] -= force_vecs
        
        # Adaptive step size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < tol:
            break
        step = min(0.1, tol / max_force)
        
        # Update
        points += step * forces
        
        # Clamp to boundary
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
    
    return points

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # Hex lattice coordinates:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * j * sqrt(3) / 2
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
        # If we didn't generate enough, shrink spacing slightly and retry
        # This is a fallback, the estimate should be close
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
        # Sort by distance to origin to pick the most central ones?
        # Or just pick the first n? Picking central ones might be better for stability.
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    points = []
    # Estimate ring structure
    # Number of rings ~ sqrt(n)
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    # Distribute points among rings
    # Ring k has roughly proportional to circumference
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            # Approximate count based on circumference
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    # Adjust to match total
    diff = n - sum(ring_counts)
    if diff > 0:
        # Add extra to outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        # Remove from outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            remove = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= remove
            diff += remove
            if diff >= 0:
                break
    
    # Ensure we have exactly n points
    while sum(ring_counts) > n:
        for i in range(len(ring_counts)-1, -1, -1):
            if ring_counts[i] > 1:
                ring_counts[i] -= 1
                break
    while sum(ring_counts) < n:
        ring_counts[-1] += 1
    
    # Place points
    idx = 0
    for k, count in enumerate(ring_counts):
        if k == 0:
            if count >= 1:
                points.append([0.0, 0.0])
                idx += 1
        else:
            radius = k * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
            # Phase offset to stagger rings
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
        # Fill remaining with random points
        while len(points) < n:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x*x + y*y <= 1.0:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    
    # Add small random perturbation
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
    best_points = None
    best_score = -np.inf
    
    for init_points in inits:
        if time.time() > deadline - 20:
            break
        
        points = init_points
        score = _compute_score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    # Optimization using L-BFGS-B with smooth soft-min objective
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pair distances efficiently
        i_idx, j_idx = iu
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # To avoid overflow in exp, we use the log-sum-exp trick.
        # Let z = -beta * c.
        # soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        # Max values for stability
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        # exp terms
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # log_sum_exp = max_z + log(sum_exp)
        log_sum_exp = max_z + np.log(sum_exp)
        
        # Objective value: -soft_min = 1/beta * log_sum_exp
        val = log_sum_exp / beta
        
        # Gradient
        # Weights
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions
        # d(pair_dist)/dp_i = (p_i - p_j) / (2 * dist)
        # d(-beta * pair_dist)/dp_i = -beta * (p_i - p_j) / (2 * dist)
        # soft_min gradient: sum(w_k * d(-beta*c_k)/dp) / beta = sum(w_k * d(-c_k)/dp)
        # So: grad_pair_i = sum over j of w_ij * (p_j - p_i) / (2 * pair_dist_ij)
        
        # Avoid division by zero
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        
        # Coefficient for each pair: w / (2 * pair_dist)
        coeff = w_pairs / (2.0 * safe_pair_dists)
        
        # Accumulate gradients
        # For point i: sum of coeff * (p_j - p_i) for all j > i
        # For point j: sum of coeff * (p_i - p_j) for all i < j
        np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
        
        # Boundary contributions
        # d(-bound_dist)/dp = d(-1 + norm)/dp = p / norm
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad

    # Optimization loop with increasing beta - extended schedule
    betas = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0, 100000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        try:
            # Use a tighter tolerance for higher betas
            if beta >= 20000:
                maxiter = 300
                gtol = 1e-12
                ftol = 1e-15
                maxls = 60
            elif beta >= 1000:
                maxiter = 400
                gtol = 1e-10
                ftol = 1e-13
                maxls = 50
            elif beta >= 100:
                maxiter = 300
                gtol = 1e-9
                ftol = 1e-12
                maxls = 40
            else:
                maxiter = 200
                gtol = 1e-8
                ftol = 1e-11
                maxls = 30
            
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'gtol': gtol,
                    'ftol': ftol,
                    'maxls': maxls,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 10:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                current_points = best_points
                
        except Exception:
            pass
    
    # Force-based refinement to handle hard constraints more directly
    if time.time() < deadline - 5:
        try:
            refined = _force_based_refine(best_points, max_iter=8000, tol=1e-12, deadline=deadline - 3)
            refined_score = _compute_score(refined)
            if refined_score > best_score:
                best_points = refined
                best_score = refined_score
        except Exception:
            pass
    
    # Final high-precision polish with active set if time permits
    if time.time() < deadline - 3:
        try:
            # Identify active constraints (tightest pairs and boundary)
            current_r = best_score
            p = best_points
            iu = np.triu_indices(N, k=1)
            i_idx, j_idx = iu
            vecs = p[i_idx] - p[j_idx]
            pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
            norms = np.linalg.norm(p, axis=1)
            bound_dists = 1.0 - norms
            
            # Active set: constraints within 1.5% of current minimum
            active_pairs = np.where(pair_dists < current_r * 1.015)[0]
            active_bounds = np.where(bound_dists < current_r * 1.015)[0]
            
            if len(active_pairs) + len(active_bounds) > 0:
                # Optimize only on active constraints with very tight tolerance
                def active_objective_and_grad(x):
                    pp = x.reshape(-1, 2)
                    
                    # Only compute for active constraints
                    if len(active_pairs) > 0:
                        a_vecs = pp[i_idx[active_pairs]] - pp[j_idx[active_pairs]]
                        a_pair_dists = np.sqrt(np.sum(a_vecs**2, axis=1)) / 2.0
                    else:
                        a_pair_dists = np.array([])
                        a_vecs = np.empty((0, 2))
                    
                    if len(active_bounds) > 0:
                        a_norms = np.linalg.norm(pp[active_bounds], axis=1)
                        a_bound_dists = 1.0 - a_norms
                    else:
                        a_bound_dists = np.array([])
                    
                    # Soft-min over active constraints only
                    beta_active = 50000.0
                    z_pairs = -beta_active * a_pair_dists if len(a_pair_dists) > 0 else np.array([-np.inf])
                    z_bounds = -beta_active * a_bound_dists if len(a_bound_dists) > 0 else np.array([-np.inf])
                    
                    all_z = np.concatenate([z_pairs, z_bounds])
                    max_z = np.max(all_z)
                    exp_all = np.exp(all_z - max_z)
                    sum_exp = np.sum(exp_all)
                    log_sum_exp = max_z + np.log(sum_exp)
                    val = log_sum_exp / beta_active
                    
                    grad = np.zeros_like(pp)
                    w_all = exp_all / sum_exp
                    
                    if len(active_pairs) > 0:
                        w_pairs = w_all[:len(a_pair_dists)]
                        safe_pd = np.where(a_pair_dists > 1e-12, a_pair_dists, 1.0)
                        coeff = w_pairs / (2.0 * safe_pd)
                        np.add.at(grad, i_idx[active_pairs], -coeff[:, np.newaxis] * a_vecs)
                        np.add.at(grad, j_idx[active_pairs], coeff[:, np.newaxis] * a_vecs)
                    
                    if len(active_bounds) > 0:
                        w_bounds = w_all[len(a_pair_dists):]
                        safe_bn = np.where(a_norms > 1e-12, a_norms, 1.0)
                        bound_grad_vecs = pp[active_bounds] / safe_bn[:, np.newaxis]
                        grad[active_bounds] += w_bounds[:, np.newaxis] * bound_grad_vecs
                    
                    return val, grad
                
                res = minimize(
                    lambda x: active_objective_and_grad(x)[0],
                    best_points.flatten(),
                    jac=lambda x: active_objective_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'gtol': 1e-14,
                        'ftol': 1e-16,
                        'maxls': 80,
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

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
    
    # Final score
    best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
