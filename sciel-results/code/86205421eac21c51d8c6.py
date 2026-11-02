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

def _active_set_refine(points, tol=1e-6, max_iter=200):
    """
    Refine only the active constraints (tightest pairs and boundary contacts).
    This is a local Newton-like step on the active set.
    """
    n = len(points)
    points = points.copy()
    
    for iteration in range(max_iter):
        # Compute current score
        score = _compute_score(points)
        if score <= 0:
            break
        
        # Find active constraints: pairs with dist/2 close to score, boundary with 1-|p| close to score
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        
        # Active pairs: dist/2 within tol of score
        active_pairs = np.argwhere(pair_dists < score + tol)
        # Remove self-pairs
        mask = active_pairs[:, 0] != active_pairs[:, 1]
        active_pairs = active_pairs[mask]
        # Only keep unique pairs (i < j)
        iu_mask = active_pairs[:, 0] < active_pairs[:, 1]
        active_pairs = active_pairs[iu_mask]
        
        # Active boundary: 1-|p| within tol of score
        active_boundary = np.where(boundary_dists < score + tol)[0]
        
        n_active_pairs = len(active_pairs)
        n_active_boundary = len(active_boundary)
        n_active = n_active_pairs + n_active_boundary
        
        if n_active == 0:
            break
        
        # Build linear system: for each active constraint, we want to increase the minimum distance
        # For pair (i,j): we want to move i and j apart
        # For boundary i: we want to move i inward
        
        # Use a simple projected gradient approach on active set
        # Compute gradient of the active constraints only
        
        grad = np.zeros_like(points)
        
        for (i, j) in active_pairs:
            vec = points[i] - points[j]
            d = np.linalg.norm(vec)
            if d < 1e-12:
                continue
            # Push apart: gradient of dist/2 wrt point i is vec/(2*d), wrt point j is -vec/(2*d)
            # We want to maximize min, so we push in direction that increases this distance
            coeff = 1.0 / (2.0 * d)
            grad[i] += coeff * vec
            grad[j] -= coeff * vec
        
        for i in active_boundary:
            norm = np.linalg.norm(points[i])
            if norm < 1e-12:
                continue
            # Push inward: gradient of (1-|p|) wrt p is -p/|p|
            # We want to maximize, so move in direction -grad = p/|p|... wait
            # d(1-|p|)/dp = -p/|p|
            # To increase 1-|p|, move in direction -(-p/|p|) = p/|p|? No.
            # We want to increase the constraint value, so move opposite to gradient of constraint
            # Actually, we're maximizing min of constraints. For active boundary constraint g_i = 1-|p_i|,
            # we want to increase g_i, so move in direction -grad_g_i = p_i/|p_i|
            # But that moves outward! That's wrong.
            # Let me reconsider: we want to maximize r such that all constraints >= r.
            # For the active set at current r, we want to find a direction that increases all active constraints.
            # For pair (i,j): direction to increase dist is along (p_i - p_j)/|p_i-p_j| for i, opposite for j
            # For boundary i: direction to increase (1-|p_i|) is -p_i/|p_i| (inward)
            
            grad[i] -= points[i] / norm
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Line search
        alpha = 1e-4
        for _ in range(20):
            new_points = points + alpha * grad
            new_score = _compute_score(new_points)
            if new_score > score:
                points = new_points
                break
            alpha *= 0.5
        else:
            break
    
    return points

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    Uses optimal spacing estimate and selects points to maximize minimum distance.
    """
    # Better spacing estimate: for N points in unit circle with hex packing,
    # the optimal radius r satisfies approximately: N * (2r)^2 * sqrt(3)/2 ≈ pi * (1-r)^2
    # But a simpler estimate: s = sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate hex lattice points
    max_k = int(1.0 / (s * 0.4)) + 3
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.99:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.95
        max_k = int(1.0 / (s * 0.4)) + 4
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.99:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Select n points that maximize minimum pairwise distance
        # Greedy: start with center, then add point farthest from existing
        radii = np.linalg.norm(candidates, axis=1)
        center_idx = np.argmin(radii)
        selected = [center_idx]
        remaining = list(range(len(candidates)))
        remaining.remove(center_idx)
        
        while len(selected) < n and remaining:
            # For each remaining candidate, compute min distance to selected
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                min_d = min(np.linalg.norm(candidates[idx] - candidates[s]) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        points = candidates[selected]
    else:
        # Fallback: use all candidates + random fill
        points = candidates.copy()
        n_needed = n - len(points)
        for _ in range(n_needed):
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
    
    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 5e-5, points.shape)
    
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
    deadline = start_time + 147.0  # 3s safety margin
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
    best_points = None
    best_score = -np.inf
    
    for init_points in inits:
        if time.time() > deadline - 30:
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

    # Optimization loop with increasing beta - adaptive schedule
    betas = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            # Use a tighter tolerance for higher betas
            if beta >= 12000:
                maxiter = 600
                gtol = 1e-13
                ftol = 1e-16
                maxls = 80
            elif beta >= 3000:
                maxiter = 500
                gtol = 1e-12
                ftol = 1e-15
                maxls = 60
            elif beta >= 500:
                maxiter = 450
                gtol = 1e-11
                ftol = 1e-14
                maxls = 50
            elif beta >= 100:
                maxiter = 400
                gtol = 1e-10
                ftol = 1e-13
                maxls = 40
            else:
                maxiter = 300
                gtol = 1e-9
                ftol = 1e-12
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

    # Active set refinement
    if time.time() < deadline - 5:
        try:
            refined = _active_set_refine(best_points, tol=1e-5, max_iter=100)
            refined_score = _compute_score(refined)
            if refined_score > best_score:
                best_points = refined
                best_score = refined_score
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 3:
        try:
            final_beta = 50000.0
            res = minimize(
                lambda x: objective_and_grad(x, final_beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, final_beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 800,
                    'gtol': 1e-14,
                    'ftol': 1e-17,
                    'maxls': 100,
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

    # Final active set refinement
    if time.time() < deadline - 1:
        try:
            refined = _active_set_refine(best_points, tol=1e-7, max_iter=50)
            refined_score = _compute_score(refined)
            if refined_score > best_score:
                best_points = refined
                best_score = refined_score
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
