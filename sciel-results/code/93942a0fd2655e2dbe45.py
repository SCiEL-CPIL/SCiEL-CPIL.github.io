"""Improved program for packing 300 circles in a unit circle using L-BFGS-B optimization."""
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

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
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
            # Approximate count based on circumference
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
    """
    Construct 300 equal circles inside a unit circle.
    Returns array of shape (N, 2) with centers.
    """
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    # Precompute upper triangle indices for efficiency
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    n_pairs = len(i_idx)
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
    for init_points in inits:
        if time.time() > deadline - 10:
            break
            
        points = init_points
        score = _compute_score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
        
    # Define objective and gradient for L-BFGS-B
    # We minimize -soft_min(constraints)
    # soft_min(c) = -1/beta * log(sum(exp(-beta * c)))
    # obj = 1/beta * log(sum(exp(-beta * c)))
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Pairwise distances
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1) + 1e-12) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Combine all constraints
        # c = [pair_dists, bound_dists]
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        # Log-sum-exp trick for numerical stability
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        log_sum_exp = max_z + np.log(sum_exp)
        
        # Objective value: 1/beta * log_sum_exp
        val = log_sum_exp / beta
        
        # Gradient
        # w_pairs = exp_pairs / sum_exp
        # w_bounds = exp_bounds / sum_exp
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad_flat = np.zeros_like(x)
        grad = grad_flat.reshape(-1, 2)
        
        # Pairwise gradients
        # d/dp_i (|p_i - p_j|/2) = (p_i - p_j) / |p_i - p_j| * 1/2
        # d/dp_j (|p_i - p_j|/2) = (p_j - p_i) / |p_i - p_j| * 1/2
        # Gradient of obj = 1/beta * sum(w_k * d(c_k)/dx)
        # Since obj = 1/beta * LSE, d(obj)/dx = 1/beta * sum(exp(z_k)/sum_exp * d(-beta*c_k)/dx)
        # = 1/beta * sum(w_k * (-beta * d(c_k)/dx))
        # = - sum(w_k * d(c_k)/dx)
        
        # So we compute - sum(w_k * grad_c_k)
        
        # For pairs
        # grad_c_k for p_i is 0.5 * (p_i - p_j) / d_k
        # grad_c_k for p_j is 0.5 * (p_j - p_i) / d_k
        
        # Accumulate gradients
        # We need to add contributions to grad[i] and grad[j]
        # grad[i] -= w_k * 0.5 * (p[i] - p[j]) / d_k
        # grad[j] -= w_k * 0.5 * (p[j] - p[i]) / d_k
        
        # Vectorized accumulation
        # This is tricky to vectorize perfectly without loops or advanced indexing
        # For N=300, n_pairs ~ 45k. A loop might be okay if optimized, but let's try to vectorize.
        
        # Actually, let's just use np.add.at for accumulation if needed, or a loop.
        # Given the deadline, a loop over 45k items in Python might be slow.
        # Let's try to vectorize the gradient calculation.
        
        # vecs = p[i_idx] - p[j_idx]
        # d_k = |vecs| / 2
        # unit_vecs = vecs / |vecs|
        # grad_vecs = w_pairs[:, None] * unit_vecs * 0.5
        
        # We want:
        # grad[i_idx] -= grad_vecs
        # grad[j_idx] += grad_vecs
        
        # Using np.add.at
        np.add.at(grad, i_idx, -w_pairs[:, None] * vecs / (2.0 * np.sqrt(np.sum(vecs**2, axis=1) + 1e-12) + 1e-12) * 0.5 * 2.0) 
        # Wait, d(c_k)/dx = 0.5 * (p_i - p_j) / d_k.
        # So - sum w_k * d(c_k)/dx = - sum w_k * 0.5 * (p_i - p_j) / d_k
        # The term (p_i - p_j) / d_k is the unit vector.
        # So we add -0.5 * w_k * unit_vec to grad[i]
        # And +0.5 * w_k * unit_vec to grad[j]
        
        # Let's redo the vectorization carefully.
        # d_k is distance/2.
        # dists_full = np.sqrt(np.sum(vecs**2, axis=1))
        # unit_vecs = vecs / dists_full[:, None]
        # grad_contrib = w_pairs[:, None] * unit_vecs * 0.5
        
        # np.add.at(grad, i_idx, -grad_contrib)
        # np.add.at(grad, j_idx, grad_contrib)
        
        # Boundary gradients
        # c_k = 1 - |p_i|
        # d(c_k)/dx = -p_i / |p_i|
        # - sum w_k * d(c_k)/dx = sum w_k * p_i / |p_i|
        
        # grad[i] += w_bounds[i] * p[i] / |p[i]|
        
        norms_safe = np.maximum(norms, 1e-12)
        bound_grad = w_bounds[:, None] * p / norms_safe[:, None]
        grad += bound_grad
        
        return val, grad_flat

    # Optimization loop
    # We will run L-BFGS-B for a few iterations, increasing beta (sharpness)
    
    current_points = best_points
    current_score = best_score
    
    beta = 50.0
    max_iter = 50
    
    for opt_round in range(3):
        if time.time() > deadline - 5:
            break
            
        # Run L-BFGS-B
        # x0 is flattened points
        x0 = current_points.flatten()
        
        # We use a fixed beta for this round, then increase it
        res = minimize(
            fun=lambda x: objective_and_grad(x, beta)[0],
            x0=x0,
            jac=lambda x: objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < objective_and_grad(x0, beta)[0]:
            new_points = res.x.reshape(-1, 2)
            # Project inside unit circle just in case
            radii = np.linalg.norm(new_points, axis=1)
            mask = radii > 1.0
            if np.any(mask):
                new_points[mask] = new_points[mask] / radii[mask, None] * 0.999999
                
            new_score = _compute_score(new_points)
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                best_points = new_points.copy()
                best_score = new_score
        
        # Increase beta for next round to sharpen the minimum
        beta *= 2.0
        max_iter = max(10, max_iter // 2)
        
    return best_points
# EVOLVE-BLOCK-END
