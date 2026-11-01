"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization with vectorized gradients."""
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

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initialize
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pair distances efficiently
        vecs = p[i_idx] - p[j_idx]
        pair_dists_sq = np.sum(vecs**2, axis=1)
        pair_dists = np.sqrt(pair_dists_sq) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft-min calculation
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
        
        # Pair contributions
        # d(pair_dist)/dp_i = (p_i - p_j) / (2 * dist)
        # We want -d(soft_min)/dp = - sum w_i * d(-c_i)/dp = sum w_i * d(c_i)/dp
        # Wait, objective is soft_min. We want to maximize it.
        # Minimize -soft_min.
        # grad of -soft_min = - sum w_i * grad(c_i)
        # c_i = dist/2. grad(c_i) wrt p_i = (p_i - p_j) / (2 * dist) = vecs / (2 * dist)
        # Note: vecs = p_i - p_j. So grad(c_i) wrt p_i = vecs / (2*dist).
        # grad(c_i) wrt p_j = -vecs / (2*dist).
        
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        
        # coeff = w_pairs / (2 * pair_dist)
        # grad_i += coeff * vecs
        # grad_j -= coeff * vecs
        
        coeff = w_pairs / (2.0 * safe_pair_dists)
        
        # Vectorized accumulation
        # This is the slow part in the original. We can try to be smarter or just rely on numpy's speed.
        # np.add.at is generally okay but can be slow.
        # Alternative: use scipy.sparse or just accept the cost if beta schedule is shorter.
        
        # Let's try to optimize the accumulation slightly by ensuring contiguous memory if possible, 
        # but np.add.at is the standard way.
        
        grad_i_contrib = coeff[:, np.newaxis] * vecs
        grad_j_contrib = -grad_i_contrib
        
        np.add.at(grad, i_idx, grad_i_contrib)
        np.add.at(grad, j_idx, grad_j_contrib)
        
        # Boundary contributions
        # c = 1 - norm. grad(c) = -p/norm.
        # grad of -c = p/norm.
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad

    # Optimization loop with annealed beta
    # Start with low beta to find a good global structure, then increase to refine.
    # The "annealed" schedule suggests smoother transitions.
    
    current_points = best_points.copy()
    
    # Define a list of betas. We will use fewer but larger steps to save time,
    # but with tighter tolerances at the end.
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0, 200000.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            # Adjust tolerances based on beta
            if beta >= 10000:
                maxiter = 300
                gtol = 1e-12
                ftol = 1e-15
                maxls = 50
            elif beta >= 1000:
                maxiter = 400
                gtol = 1e-11
                ftol = 1e-14
                maxls = 40
            elif beta >= 100:
                maxiter = 500
                gtol = 1e-10
                ftol = 1e-13
                maxls = 30
            else:
                maxiter = 600
                gtol = 1e-9
                ftol = 1e-12
                maxls = 20
            
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
            
            if time.time() > deadline - 5:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                # If it didn't improve, stick with the best known
                current_points = best_points
                
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 2:
        try:
            final_beta = 1000000.0
            res = minimize(
                lambda x: objective_and_grad(x, final_beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, final_beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-13,
                    'ftol': 1e-16,
                    'maxls': 30,
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
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
