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

def _hex_lattice_init(n, seed=42):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    np.random.seed(seed)
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
        # Sort by distance to origin to pick the most central ones
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

def _ring_init(n, seed=123):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    np.random.seed(seed)
    points = []
    # Estimate ring structure
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    # Distribute points among rings
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    # Adjust to match total
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
    
    # Add small random perturbation
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute upper triangle indices for efficiency
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pair distances efficiently
        i_idx, j_idx = iu
        vecs = p[i_idx] - p[j_idx]
        pair_dists_sq = np.sum(vecs**2, axis=1)
        pair_dists = np.sqrt(pair_dists_sq) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # To avoid overflow in exp, we use the log-sum-exp trick.
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
        
        # Gradient
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Gradient from pair distances
        # d/dp_i (1/2 |p_i - p_j|) = (p_i - p_j) / (2 * |p_i - p_j|) * 0.5
        # Actually, soft_min of c where c_ij = |p_i - p_j|/2
        # grad_val = -sum w_ij * grad_c_ij
        # grad_c_ij w.r.t p_i = (p_i - p_j) / (2 * |p_i - p_j|) * 0.5? No.
        # c = dist/2. d(c)/d(p_i) = (p_i - p_j) / (2 * dist)
        # So term is w_ij * (p_i - p_j) / (2 * dist)
        # But we want to MINIMIZE the soft min, so we maximize the negative.
        # The objective is val = -soft_min. We want to minimize val.
        # grad val = - grad soft_min
        # soft_min = -1/beta * log(sum exp(-beta c))
        # d(soft_min)/d(p) = sum (exp(-beta c) / sum_exp) * (-beta) * d(-c)/dp ?
        # Let's use the standard formula:
        # soft_min(c) = -1/beta * log(sum exp(-beta c_k))
        # d(soft_min)/dx = sum_k (exp(-beta c_k) / sum_exp) * d(-c_k)/dx * (-1) ?
        # Let S = sum exp(-beta c_k).
        # d(-1/beta log S)/dx = -1/beta * (1/S) * dS/dx
        # dS/dx = sum exp(-beta c_k) * (-beta) * dc_k/dx
        # So d(soft_min)/dx = -1/beta * 1/S * sum exp(-beta c_k) * (-beta) * dc_k/dx
        # = 1/S * sum exp(-beta c_k) * dc_k/dx
        # = sum w_k * dc_k/dx
        # We want to minimize -soft_min, so grad_obj = - sum w_k * dc_k/dx
        
        # For pair (i, j): c_ij = |p_i - p_j| / 2
        # dc_ij / dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
        # dc_ij / dp_j = (p_j - p_i) / (2 * |p_i - p_j|)
        
        # Vectorized accumulation
        # For each pair (i, j) with weight w:
        # grad[i] += w * (p[i] - p[j]) / (2 * dist_ij)
        # grad[j] += w * (p[j] - p[i]) / (2 * dist_ij)
        
        # Note: dist here is |p_i - p_j|, not c_ij.
        # c_ij = dist/2.
        # dc/dp_i = (p_i - p_j) / (2 * dist)
        
        safe_dists = np.where(pair_dists_sq > 1e-12, pair_dists_sq, 1.0)
        inv_dists = 1.0 / np.sqrt(safe_dists)
        
        # Factor for gradient: w * (p_i - p_j) / (2 * dist)
        # We accumulate into grad
        # grad[i] += sum_j w_ij * (p_i - p_j) * inv_dist_ij * 0.5
        # grad[j] -= sum_i w_ij * (p_i - p_j) * inv_dist_ij * 0.5
        
        # Using np.add.at
        # vecs = p[i_idx] - p[j_idx]
        # factor = w_pairs * inv_dists * 0.5
        # contrib = vecs * factor[:, np.newaxis]
        
        factor = w_pairs * inv_dists * 0.5
        contrib = vecs * factor[:, np.newaxis]
        
        np.add.at(grad, i_idx, contrib)
        np.add.at(grad, j_idx, -contrib)
        
        # Boundary gradient
        # c_b = 1 - |p|
        # dc_b / dp = -p / |p|
        # grad[i] += w_b_i * (-p_i / |p_i|)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        b_factor = w_bounds / safe_norms
        b_contrib = -p * b_factor[:, np.newaxis]
        
        grad += b_contrib
        
        # The objective is val = -soft_min.
        # We computed grad(soft_min).
        # So grad(val) = -grad(soft_min)
        grad = -grad
        
        return val, grad

    best_points = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N, seed=42), _ring_init(N, seed=123)]
    
    for init_points in inits:
        if time.time() > deadline - 30:
            break
        
        points = init_points
        score = _compute_score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N, seed=42)
        best_score = _compute_score(best_points)
    
    # Optimization using L-BFGS-B with smooth soft-min objective
    # We minimize -soft_min, which is equivalent to maximizing soft_min.
    # As beta increases, soft_min approximates min better.
    # We use an annealing schedule for beta.
    
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        x0 = current_points.flatten()
        
        res = minimize(
            objective_and_grad,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.success or res.fun < objective_and_grad(x0, beta)[0]:
            new_points = res.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            
            if new_score > best_score:
                best_points = new_points.copy()
                best_score = new_score
                current_points = new_points.copy()
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Final polish with a very high beta if time permits
    if time.time() < deadline - 5:
        beta_final = 1000.0
        x0 = best_points.flatten()
        
        res = minimize(
            objective_and_grad,
            x0,
            args=(beta_final,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 30}
        )
        
        if res.success:
            new_points = res.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            
            if new_score > best_score:
                best_points = new_points.copy()
                best_score = new_score
    
    return best_points

# EVOLVE-BLOCK-END
