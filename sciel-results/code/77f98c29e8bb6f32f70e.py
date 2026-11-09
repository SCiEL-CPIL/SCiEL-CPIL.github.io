import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _softmin_grad(values, beta):
    """
    Compute the softmin of values and its gradient with respect to values.
    values: array of shape (M,)
    Returns: softmin_value, weights (shape (M,))
    """
    # softmin(v) = -1/beta * log(sum(exp(-beta * v)))
    # For numerical stability, subtract max
    neg_beta_v = -beta * values
    max_neg_beta_v = np.max(neg_beta_v)
    exp_vals = np.exp(neg_beta_v - max_neg_beta_v)
    sum_exp = np.sum(exp_vals)
    
    # Softmin value
    softmin_val = -(max_neg_beta_v / beta + np.log(sum_exp) / beta)
    
    # Weights: w_i = exp(-beta * v_i) / sum(exp(-beta * v))
    # This is the derivative of softmin w.r.t v_i
    weights = exp_vals / sum_exp
    
    return softmin_val, weights

def _compute_score(points):
    """
    Compute the actual score (max radius) for a set of points.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Boundary constraint
    norms = np.linalg.norm(points, axis=1)
    boundary_r = np.min(1.0 - norms)
    
    if boundary_r < 0:
        return 0.0
        
    # Pairwise distances
    # Using pdist for speed in scoring
    from scipy.spatial.distance import pdist
    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _generate_hexagonal_lattice(spacing, n_target):
    """
    Generate a hexagonal lattice of points inside the unit disk.
    Select the n_target points closest to the origin.
    """
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    
    if len(pts) < n_target:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:n_target]
    return pts[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 150s limit, leave 10s buffer
    
    # 1. Initialization
    # Theoretical best radius is ~ 0.05315. 
    # Hex lattice spacing s = 2*r.
    # Let's try a few spacings around the expected optimal value.
    best_points = None
    best_score = 0.0
    
    # Spacings corresponding to r in [0.050, 0.055]
    spacings = [0.100, 0.104, 0.106, 0.108, 0.110]
    
    for spacing in spacings:
        if time.time() > deadline - 60.0:
            break
        pts = _generate_hexagonal_lattice(spacing, N)
        if pts is not None:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
    if best_points is None:
        # Fallback: random
        best_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
        best_score = _compute_score(best_points)
        
    # 2. Optimization using SLSQP with Soft-Min Objective
    # We want to maximize min( min_{i<j} d_ij/2, min_i (1-||c_i||) )
    # We approximate min with softmin.
    # Objective: Minimize -softmin( values )
    # Values: [d_ij/2 for i<j, 1-||c_i|| for i]
    
    beta = 50.0  # Sharpness of softmin. Higher is better approximation to min.
    
    x_current = best_points.flatten().copy()
    
    def objective_and_gradient(x):
        points = x.reshape(-1, 2)
        
        # Compute all pairwise distances
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise values: d_ij / 2
        # We only need upper triangle for the set of values, but for gradient
        # calculation, it's easier to work with the full matrix and mask.
        # However, softmin sum is over unique pairs.
        # Let's extract upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Boundary values: 1 - ||c_i||
        norms = np.linalg.norm(points, axis=1)
        boundary_vals = 1.0 - norms
        
        # Combine values
        all_vals = np.concatenate([pair_dists, boundary_vals])
        
        # Compute softmin and weights
        softmin_val, weights = _softmin_grad(all_vals, beta)
        
        # Objective value is -softmin_val
        obj_val = -softmin_val
        
        # Compute gradient
        # grad_x = - sum_k w_k * grad_x(v_k)
        # w_k are the weights.
        # Split weights back into pair and boundary parts
        n_pairs = len(pair_dists)
        w_pairs = weights[:n_pairs]
        w_bounds = weights[n_pairs:]
        
        grad = np.zeros_like(points)
        
        # Gradient of pair terms
        # v_ij = ||x_i - x_j|| / 2
        # grad_{x_i} v_ij = (x_i - x_j) / (2 * ||x_i - x_j||)
        # grad_{x_j} v_ij = (x_j - x_i) / (2 * ||x_i - x_j||)
        
        # We need to accumulate contributions from all pairs.
        # This is O(N^2).
        # For each pair (i, j) with weight w:
        # grad[i] += -w * (x_i - x_j) / (2 * d_ij)
        # grad[j] += -w * (x_j - x_i) / (2 * d_ij)
        
        # Vectorized approach:
        # Create a matrix of gradients for pairs?
        # dists is N x N.
        # safe_dists = np.where(dists < 1e-8, 1e-8, dists)
        # factor_matrix = np.zeros((N, N))
        # factor_matrix[i_idx, j_idx] = w_pairs
        # factor_matrix[j_idx, i_idx] = w_pairs # Symmetric
        
        # grad_from_pairs[i] = - sum_j factor_matrix[i, j] * (x_i - x_j) / (2 * d_ij)
        # Note: d_ij = dists[i, j]
        
        # Let's compute the term (x_i - x_j) / (2 * d_ij)
        # This is diff[i, j] / (2 * dists[i, j])
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-8, 1e-8, dists)
        
        # weight_matrix is symmetric
        weight_matrix = np.zeros((N, N))
        weight_matrix[i_idx, j_idx] = w_pairs
        weight_matrix[j_idx, i_idx] = w_pairs
        
        # grad_pair[i] = - sum_j weight_matrix[i, j] * diff[i, j] / (2 * dists_safe[i, j])
        # diff[i, j] = x_i - x_j
        
        # We can compute this as:
        # term = weight_matrix[:, :, np.newaxis] * diff / (2 * dists_safe[:, :, np.newaxis])
        # grad_pair = - np.sum(term, axis=1)
        
        # This creates a N x N x 2 array, which is 300*300*2*8 = 1.44 MB. Acceptable.
        
        term_pairs = weight_matrix[:, :, np.newaxis] * diff / (2.0 * dists_safe[:, :, np.newaxis])
        grad_pairs = -np.sum(term_pairs, axis=1)
        
        # Gradient of boundary terms
        # v_i = 1 - ||x_i||
        # grad_{x_i} v_i = - x_i / ||x_i||
        # grad_bound[i] = - w_bounds[i] * (- x_i / ||x_i||) = w_bounds[i] * x_i / ||x_i||
        
        norms_safe = np.where(norms < 1e-8, 1e-8, norms)
        grad_bounds = w_bounds[:, np.newaxis] * points / norms_safe[:, np.newaxis]
        
        total_grad = (grad_pairs + grad_bounds).flatten()
        
        return obj_val, total_grad

    # Constraints for SLSQP:
    # We enforce that points stay inside the unit circle (radius 1).
    # This is a hard constraint to prevent the optimizer from pushing points
    # far outside where the soft-min penalty might be weak or numerical issues arise.
    # c_i(x) = 1 - ||x_i|| >= 0
    
    def constraint_factory(i):
        def c(x):
            points = x.reshape(-1, 2)
            return 1.0 - np.linalg.norm(points[i])
        return c

    # SLSQP requires a list of constraint dicts
    constraints = []
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': constraint_factory(i)
        })
    
    # Run SLSQP
    # maxiter: limit iterations to ensure we finish in time.
    # ftol: tolerance for function value.
    
    try:
        result = minimize(
            objective_and_gradient, 
            x_current, 
            method='SLSQP', 
            jac=True, 
            constraints=constraints, 
            options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success or result.fun < 0: # result.fun is -softmin, so smaller is better (more negative means larger softmin)
            new_points = result.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
    except Exception as e:
        # If SLSQP fails, we still have the initial best_points
        pass

    # If we have time, we could try a second round with higher beta or different start,
    # but SLSQP is expensive. Let's stick to one robust run.
    
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)):
         # Fallback to a simple ring if something went wrong
         angles = 2 * np.pi * np.arange(N) / N
         best_points = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
         best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
