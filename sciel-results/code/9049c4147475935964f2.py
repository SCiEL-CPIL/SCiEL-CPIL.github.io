"""Combined program: hexagonal lattice init + soft-min optimization with epigraph variable."""
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
    deadline = start_time + 145.0  # 5s safety margin
    
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
        
        # Pairwise contributions
        # d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij)
        # d/dp_j (d_ij/2) = (p_j - p_i) / (2 * d_ij)
        # Note: pair_dists is d/2, so d = 2*pair_dists
        d_pairs = 2.0 * pair_dists
        mask = d_pairs > 1e-12
        w_pairs_safe = np.where(mask, w_pairs, 0.0)
        d_pairs_safe = np.where(mask, d_pairs, 1.0)
        
        unit_vecs = vecs / d_pairs_safe[:, np.newaxis]
        coeffs = w_pairs_safe / 2.0
        
        np.add.at(grad, i_idx, coeffs[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -coeffs[:, np.newaxis] * unit_vecs)
        
        # Boundary contributions
        # Constraint: 1 - |p_i|.
        # Grad: -p_i / |p_i|.
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        bound_grads = -p / norms_safe[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grads
        
        return val, -grad
    
    # Annealing loop
    # Start with low beta (smooth), end with high beta (sharp)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_points = best_points.flatten()
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-6,
                    'ftol': 1e-10,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() < deadline:
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    current_points = candidate.flatten()
                else:
                    # If it didn't improve the hard score, keep the previous best for next iteration
                    # or use the candidate if it's closer to the target?
                    # Usually, if hard score didn't improve, the soft min optimization might have moved
                    # away from the optimal hard min. 
                    # However, if we are annealing, we want to follow the soft min.
                    # Let's use the candidate as the start for the next (sharper) beta,
                    # but keep best_points as the one with highest hard score.
                    current_points = candidate.flatten()
                    
        except Exception as e:
            # If optimization fails, break
            break
            
    # Final polish with a very high beta (effectively hard min)
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                lambda x: objective_and_grad(x, 5000.0)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, 5000.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() < deadline:
                candidate = res_final.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    
        except Exception as e:
            pass

    # Final check
    norms = np.linalg.norm(best_points, axis=1)
    # Ensure points are inside
    # If any point is outside, scale it back? 
    # The objective penalizes going outside, so it should be fine.
    
    return best_points

# EVOLVE-BLOCK-END
