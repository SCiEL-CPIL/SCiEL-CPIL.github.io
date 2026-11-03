"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_lattice(n):
    """Generate a hexagonal lattice of points inside the unit circle."""
    # Estimate spacing
    # Area of hexagon ~ 1.1 * d^2. Total area ~ pi.
    # n * 1.1 * d^2 ~ pi => d ~ sqrt(pi / (1.1 * n))
    d = np.sqrt(np.pi / (1.1 * n))
    
    points = []
    # Generate points in a grid that covers the unit circle
    # Range of x, y
    k = int(2.0 / d) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = i * d + (j % 2) * 0.5 * d
            y = j * d * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Add random points if lattice is too sparse
        while len(points) < n:
            p = np.random.uniform(-1, 1, 2)
            if np.hypot(*p) < 1.0:
                points = np.vstack([points, p])
    elif len(points) > n:
        # Select n points closest to center
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:n]
        points = points[idx]
        
    return points

def _objective_and_grad(x, beta):
    """
    Compute the negative LogSumExp smoothed minimum distance and its gradient.
    x: array of shape (N, 2)
    beta: smoothing parameter
    """
    points = x.reshape(-1, 2)
    N = len(points)
    
    # Pairwise distances
    # To save memory and time, we can compute in chunks or use broadcasting carefully
    # For N=300, N^2 = 90,000. This is fine.
    
    # Compute all pairwise differences
    # diff[i, j] = points[i] - points[j]
    # This is 300x300x2 = 180,000 floats.
    
    # We only need i < j
    # Let's compute distances for i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = points[i_idx] - points[j_idx]
    dists = np.linalg.norm(diffs, axis=1) / 2.0
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    # Combine distances
    # We want to maximize min(dists, boundary_dists)
    # g_beta = -1/beta * log(sum(exp(-beta * d)))
    # We want to minimize -g_beta = 1/beta * log(sum(exp(-beta * d)))
    
    all_dists = np.concatenate([dists, boundary_dists])
    
    # Numerical stability: subtract max
    max_d = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_d))
    sum_exp = np.sum(exp_terms)
    
    # Objective value
    val = (1.0 / beta) * (np.log(sum_exp) - beta * max_d)
    
    # Gradient
    # d(-g_beta)/dx = - d(g_beta)/dx
    # g_beta = -1/beta log(S) where S = sum exp(-beta d_k)
    # dg_beta/dx = -1/beta * (1/S) * dS/dx
    # dS/dx = sum (-beta * exp(-beta d_k) * dd_k/dx)
    # So dg_beta/dx = sum (exp(-beta d_k)/S * dd_k/dx)
    # We want to minimize -g_beta, so grad = -dg_beta/dx = - sum (...)
    
    # Weights
    weights = exp_terms / sum_exp
    
    # Gradient of distances w.r.t points
    # grad_x = zeros((N, 2))
    grad_x = np.zeros_like(points)
    
    # Pairwise part
    # weight_k corresponds to pair (i_idx[k], j_idx[k])
    # d(d_k)/dx_i = (x_i - x_j) / (2 * |x_i - x_j|)
    # d(d_k)/dx_j = (x_j - x_i) / (2 * |x_i - x_j|)
    
    # Avoid division by zero
    norms = np.linalg.norm(diffs, axis=1)
    norms[norms < 1e-12] = 1e-12
    
    unit_vecs = diffs / norms[:, np.newaxis]
    
    # Contribution to i
    # np.add.at(grad_x, i_idx, -weights[:, np.newaxis] * unit_vecs)
    # Note: d(d_k)/dx_i = unit_vec / 2? No, dist = |diff|/2.
    # d(|diff|/2)/dx_i = (1/2) * (diff / |diff|) = unit_vec / 2.
    # So term is weights * (unit_vec / 2).
    
    # We are minimizing -g_beta.
    # grad(-g_beta) = - sum(weights * grad(d_k))
    
    # For pair k:
    # grad(d_k)/x_i = unit_vec_k / 2
    # grad(d_k)/x_j = -unit_vec_k / 2
    
    # So contribution to grad_x[i] is -weights[k] * unit_vec_k / 2
    # Contribution to grad_x[j] is -weights[k] * (-unit_vec_k / 2) = weights[k] * unit_vec_k / 2
    
    # Let's accumulate
    # Using np.add.at for efficiency
    np.add.at(grad_x, i_idx, -0.5 * weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad_x, j_idx, 0.5 * weights[:, np.newaxis] * unit_vecs)
    
    # Boundary part
    # d_k = 1 - |x_k|
    # d(d_k)/dx_k = -x_k / |x_k|
    # Contribution to grad_x[k] is -weights_boundary[k] * (-x_k / |x_k|) = weights_boundary[k] * x_k / |x_k|
    
    b_weights = weights[len(dists):]
    b_norms = np.linalg.norm(points, axis=1)
    b_norms[b_norms < 1e-12] = 1e-12
    b_unit_vecs = points / b_norms[:, np.newaxis]
    
    grad_x += b_weights[:, np.newaxis] * b_unit_vecs
    
    return val, grad_x.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_points = None
    best_radius = 0.0
    
    # Restart loop
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            points = _hex_lattice(N)
        else:
            # Perturb the previous best or generate new
            if best_points is not None:
                points = best_points + np.random.normal(0, 0.01, (N, 2))
                # Project back to unit circle if necessary? 
                # The optimizer will handle it, but let's ensure validity
                norms = np.linalg.norm(points, axis=1)
                mask = norms > 1.0
                points[mask] = points[mask] / norms[mask, np.newaxis] * 0.99
            else:
                points = _hex_lattice(N)
                
        # Optimization
        # We can increase beta over time or use a fixed high beta
        # High beta makes it sharper, but harder to optimize.
        # Let's start with beta=10 and increase? Or just use a fixed beta=20.
        # For N=300, beta=20 should be okay.
        
        x0 = points.flatten()
        
        # Define objective for scipy
        def obj(x):
            val, _ = _objective_and_grad(x, 20.0)
            return val
        
        def grad(x):
            _, g = _objective_and_grad(x, 20.0)
            return g
            
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if res.success or res.fun < 1e10: # Check if it didn't blow up
                opt_points = res.x.reshape(N, 2)
                
                # Calculate true radius
                # Pairwise
                i_idx, j_idx = np.triu_indices(N, k=1)
                diffs = opt_points[i_idx] - opt_points[j_idx]
                dists = np.linalg.norm(diffs, axis=1) / 2.0
                min_pair_dist = np.min(dists)
                
                # Boundary
                b_dists = 1.0 - np.linalg.norm(opt_points, axis=1)
                min_b_dist = np.min(b_dists)
                
                r = min(min_pair_dist, min_b_dist)
                
                if r > best_radius:
                    best_radius = r
                    best_points = opt_points
                    
        except Exception as e:
            pass # Skip failed restarts

    if best_points is None:
        # Fallback to initial lattice
        best_points = _hex_lattice(N)
        
    return best_points
# EVOLVE-BLOCK-END
