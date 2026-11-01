"""Improved program for packing 300 circles in a unit circle using radial-ring decomposition with smooth soft-min optimization."""
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
    # Estimate spacing: area per circle ~ pi * r^2, total area pi * 1^2
    # N * pi * r^2 ~ pi => r ~ 1/sqrt(N)
    # Hex spacing: s = 2*r / sqrt(3) for touching circles in hex lattice
    r_est = 1.0 / np.sqrt(n)
    s = 2.0 * r_est / np.sqrt(3.0)
    
    # Generate hex lattice points
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
        # Try slightly smaller spacing
        s *= 0.95
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
        # Fallback: random points inside circle
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9
    
    return points

def _ring_init(n, num_rings=12, ring_spread=0.95):
    """
    Initialize points in concentric rings.
    """
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    ring_radii = [ring_spread * (i + 1) / (num_rings + 1) for i in range(num_rings)]
    circumferences = [2.0 * np.pi * rr for rr in ring_radii]
    total_circ = sum(circumferences)
    
    ring_counts = []
    for i, circ in enumerate(circumferences):
        count = int(remaining * circ / total_circ)
        ring_counts.append(max(1, count))
    
    diff = remaining - sum(ring_counts)
    if diff > 0:
        for _ in range(diff):
            ring_counts[-1] += 1
    elif diff < 0:
        for i in range(len(ring_counts) - 1, -1, -1):
            if ring_counts[i] > 1 and -diff > 0:
                ring_counts[i] -= 1
                diff += 1
    
    for i, (count, rr) in enumerate(zip(ring_counts, ring_radii)):
        if count == 0:
            continue
        d_theta = 2.0 * np.pi / count
        offset = np.pi / count * (i % 2)
        for k in range(count):
            theta = k * d_theta + offset
            points.append([rr * np.cos(theta), rr * np.sin(theta)])
    
    points = np.array(points[:n])
    return points

def _soft_min_and_grad(values, beta):
    """
    Compute soft-min and its gradient weights using log-sum-exp trick.
    """
    values = np.array(values)
    z = -beta * values
    
    max_z = np.max(z)
    exp_vals = np.exp(z - max_z)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    
    return soft_min_val, weights

def _objective_and_grad(x, beta):
    """
    Compute the negative soft-min objective and its gradient.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint values
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min
    r_soft, weights = _soft_min_and_grad(all_vals, beta)
    objective = -r_soft
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Pair contributions
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    grad_obj = -grad
    
    return objective, grad_obj

def _local_optimize(points, time_limit=30.0, beta_schedule=None, max_iter=500):
    """
    Local optimization using L-BFGS-B with soft-min objective and adaptive beta.
    """
    start_time = time.time()
    
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                
        except Exception:
            continue
    
    return current_points, current_score

def _polish_with_gradient(points, time_limit=10.0):
    """
    Final polish pass with very high beta for tight convergence.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Use very high beta for precise optimization
    beta = 10000.0
    
    try:
        res = minimize(
            lambda x: _objective_and_grad(x, beta)[0],
            current_points.flatten(),
            jac=lambda x: _objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'gtol': 1e-10,
                'ftol': 1e-14,
                'maxls': 50,
                'disp': False
            }
        )
        
        candidate = res.x.reshape(-1, 2)
        candidate_score = _compute_score(candidate)
        
        if candidate_score > current_score:
            current_points = candidate
            current_score = candidate_score
            
    except Exception:
        pass
    
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initializations and pick the best
    init_candidates = []
    
    # Hex lattice
    hex_points = _hex_lattice_init(N)
    init_candidates.append(hex_points)
    
    # Ring initialization
    ring_points = _ring_init(N, 12, 0.95)
    init_candidates.append(ring_points)
    
    # Pick best initialization
    best_points = init_candidates[0]
    best_score = _compute_score(best_points)
    for pts in init_candidates[1:]:
        score = _compute_score(pts)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Phase 1: L-BFGS optimization with moderate beta
    if time.time() < deadline - 50:
        remaining_time = min(40.0, deadline - 50 - time.time())
        best_points, best_score = _local_optimize(
            best_points, 
            time_limit=remaining_time,
            beta_schedule=[50.0, 200.0, 1000.0],
            max_iter=200
        )
    
    # Phase 2: Higher beta for finer optimization
    if time.time() < deadline - 20:
        remaining_time = min(15.0, deadline - 20 - time.time())
        best_points, best_score = _local_optimize(
            best_points,
            time_limit=remaining_time,
            beta_schedule=[2000.0, 5000.0],
            max_iter=150
        )
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        remaining_time = min(10.0, deadline - 5 - time.time())
        best_points, best_score = _polish_with_gradient(
            best_points,
            time_limit=remaining_time
        )
    
    # Final validation: ensure all points are inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        for i in range(len(best_points)):
            if norms[i] > 1.0:
                best_points[i] = best_points[i] / norms[i] * (1.0 - 1e-10)
        best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
