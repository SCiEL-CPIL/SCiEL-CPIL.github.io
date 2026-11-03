"""
Combined approach: Discrete initialization + Continuous Soft-Min Gradient Optimization.

Strategy:
1. Initialize using a greedy selection on a hexagonal grid (from current program).
2. Optimize using L-BFGS-B on a soft-min objective (from inspiration).
   - The objective is the negative soft-min of all constraints (wall distances, pair distances/2).
   - This allows the optimizer to "see" near-minimal constraints and push points apart.
   - We use a schedule of increasing beta (sharpness) to refine the solution.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to return before kill

def _compute_radius(pts):
    """Compute the packing radius for given centres in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    
    # Boundary distance: min over all points and all 4 sides
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    min_boundary = np.minimum(bx.min(), by.min())
    
    # Pairwise distances
    # For N=200, cdist is O(N^2) = 40000 ops, which is very fast.
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    r = min(min_boundary, min_pair)
    return r

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the objective function and its gradient for the soft-min problem.
    
    We want to maximize the minimum of a set of values C (constraints).
    Let S_beta(C) = -1/beta * log(sum(exp(-beta * C_i)))
    We minimize -S_beta(C).
    
    x: flattened array of shape (2*N,)
    beta: sharpness parameter
    """
    pts = x.reshape((N, 2))
    
    # 1. Compute constraint values C
    # Wall constraints:
    # dist to left: x_i + 0.5
    # dist to right: 0.5 - x_i
    # dist to bottom: y_i + 0.5
    # dist to top: 0.5 - y_i
    
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair constraints:
    # dist(i, j) / 2 for i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Vectorized pair distance calculation
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    pair_vals = dists / 2.0
    
    # Combine all constraints into C
    C = np.concatenate([wall_vals, pair_vals])
    
    # 2. Compute soft-min value and weights
    # S_beta = -1/beta * log(sum(exp(-beta * C)))
    # To avoid overflow, use log-sum-exp trick:
    # log(sum(exp(y))) = max(y) + log(sum(exp(y - max(y))))
    # where y = -beta * C
    
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        # Fallback: return hard min and zero gradient if numerical issues
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights for gradient
    # w_i = exp(y_i) / sum(exp(y_j)) = exp(shifted_i) / sum_exp
    weights = exp_shifted / sum_exp
    
    # 3. Compute Gradient
    # grad_obj = - (1/beta) * sum( w_i * grad_C_i )
    # Note: C_i are the constraint values. We are minimizing -S_beta, so grad = -grad(S_beta).
    # grad(S_beta) = -1/beta * sum( w_i * grad_C_i )  <-- Wait, let's re-derive.
    # S = -1/beta * log(sum exp(-beta C))
    # dS/dx = -1/beta * (1/sum exp(-beta C)) * sum(exp(-beta C_i) * (-beta dC_i/dx))
    # dS/dx = (1/sum exp(-beta C)) * sum(exp(-beta C_i) * dC_i/dx)
    # dS/dx = sum( w_i * dC_i/dx )
    # We minimize -S, so grad_obj = - sum( w_i * dC_i/dx )
    
    grad = np.zeros_like(x)
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    # C[0:N] = x_i + 0.5 -> dC/dx_i = 1
    # C[N:2N] = 0.5 - x_i -> dC/dx_i = -1
    # C[2N:3N] = y_i + 0.5 -> dC/dy_i = 1
    # C[3N:4N] = 0.5 - y_i -> dC/dy_i = -1
    
    w_wall = weights[:4*N]
    
    # Sum contributions for x-coords
    # d/dx_i (x_i+0.5) * w_i  + d/dx_i (0.5-x_i) * w_{i+N}
    grad_pts[:, 0] += w_wall[0:N] * 1.0 + w_wall[N:2*N] * (-1.0)
    
    # Sum contributions for y-coords
    grad_pts[:, 1] += w_wall[2*N:3*N] * 1.0 + w_wall[3*N:4*N] * (-1.0)
    
    # Pair gradients
    # C[4N + k] = dist(i, j) / 2
    # Let d = dist(i, j). dC/dx_i = (x_i - x_j) / (2 * d). dC/dx_j = (x_j - x_i) / (2 * d).
    # Similarly for y.
    
    w_pairs = weights[4*N:]
    
    # We need to accumulate these into grad_pts.
    # For each pair (i, j) with weight w_k:
    # grad_pts[i] += w_k * ( (x_i - x_j)/(2d), (y_i - y_j)/(2d) )
    # grad_pts[j] += w_k * ( (x_j - x_i)/(2d), (y_j - y_i)/(2d) )
    
    # Vectorized accumulation using np.add.at
    # Calculate unit vectors scaled by 0.5
    valid = dists > 1e-10
    # Avoid division by zero
    inv_2d = np.zeros_like(dists)
    inv_2d[valid] = 0.5 / dists[valid]
    
    # Gradient components for point i in pair k
    gx_i = dx * inv_2d
    gy_i = dy * inv_2d
    
    # Weighted gradients
    wg_x_i = w_pairs * gx_i
    wg_y_i = w_pairs * gy_i
    
    # Accumulate to i
    np.add.at(grad_pts, i_idx, wg_x_i[:, None])
    np.add.at(grad_pts, i_idx, wg_y_i[:, None])
    
    # Accumulate to j (negative)
    np.add.at(grad_pts, j_idx, -wg_x_i[:, None])
    np.add.at(grad_pts, j_idx, -wg_y_i[:, None])
    
    # Total gradient is - sum(...)
    grad_pts = -grad_pts
    grad = grad_pts.flatten()
    
    return obj_val, grad

def _build_hex_grid(spacing):
    """Build a fine hexagonal grid of candidate sites within [-0.5, 0.5]^2."""
    sites = []
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    
    row = 0
    y = y_min
    while y <= y_max + 1e-9:
        offset = (dx / 2.0) if (row % 2 == 1) else 0.0
        x = x_min + offset
        while x <= x_max + 1e-9:
            if x_min - 1e-9 <= x <= x_max + 1e-9 and y_min - 1e-9 <= y <= y_max + 1e-9:
                sites.append((x, y))
            x += dx
        y += dy
        row += 1
    
    if not sites:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        sites = [(x, y) for y in ticks for x in ticks]
    
    return np.array(sites)

def _greedy_select(sites, n_select):
    """
    Greedy best-fit: repeatedly pick the site that maximizes the minimum
    distance to already-selected sites AND to the square boundary.
    """
    n = len(sites)
    if n_select >= n:
        return sites.copy()
    
    selected = []
    selected_set = set()
    
    bx = 0.5 - np.abs(sites[:, 0])
    by = 0.5 - np.abs(sites[:, 1])
    boundary_dist = np.minimum(bx, by)
    
    min_dist = np.full(n, np.inf)
    
    for step in range(n_select):
        score = np.minimum(boundary_dist, min_dist)
        best_idx = np.argmax(score)
        selected.append(sites[best_idx])
        selected_set.add(best_idx)
        
        if step < n_select - 1:
            d = np.linalg.norm(sites - sites[best_idx], axis=1)
            min_dist = np.minimum(min_dist, d)
            min_dist[list(selected_set)] = -np.inf
    
    return np.array(selected)

def construct():
    start_time = time.time()
    
    # 1. Initialization
    # Use a hex grid with spacing appropriate for N=200
    # Ideal spacing ~ 1/sqrt(200) ~ 0.07. Grid spacing 0.03 gives ~16x16=256 sites, enough.
    spacing = 0.03
    sites = _build_hex_grid(spacing)
    pts = _greedy_select(sites, N)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return pts
    
    # 2. Optimization
    # We will run L-BFGS-B with increasing beta.
    # Beta controls the sharpness of the soft-min.
    # Low beta: smooth, explores. High beta: sharp, exploits.
    
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    best_pts = pts
    best_r = _compute_radius(pts)
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        # Prepare initial guess for this beta stage
        # If it's the first stage, use the greedy result.
        # Otherwise, use the result from the previous stage.
        x0 = best_pts.flatten()
        
        # Define the objective and gradient functions for this beta
        def obj_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        # Run L-BFGS-B
        # maxiter: limit iterations to prevent hanging
        # maxfun: limit function evaluations
        # time_limit is handled by checking elapsed time in the main loop, 
        # but we should also set a reasonable maxiter/maxfun.
        
        try:
            res = minimize(
                lambda x: _soft_min_obj_and_grad(x, beta)[0],
                x0,
                jac=lambda x: _soft_min_obj_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'maxfun': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            new_pts = res.x.reshape((N, 2))
            
            # Validate result
            if np.all(np.isfinite(new_pts)):
                # Ensure points are inside the square (L-BFGS-B doesn't have bounds by default 
                # unless we add them, but our soft-min handles boundaries smoothly. 
                # However, if points go outside, the wall distance becomes negative, 
                # which the soft-min will penalize heavily. 
                # But to be safe, we can clip or just accept if radius is calculated correctly.
                # The radius calculation handles outside points by giving negative boundary dist,
                # which minimizes the score.
                
                r_new = _compute_radius(new_pts)
                if r_new > best_r:
                    best_r = r_new
                    best_pts = new_pts
                    
        except Exception as e:
            # If optimization fails, keep the previous best
            pass
        
        # Check time after each beta stage
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
    
    # 3. Final Polish (Optional)
    # If we have time, run one more high-beta optimization on the best found
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN - 5.0:
        try:
            x0 = best_pts.flatten()
            beta_final = 5000.0
            
            res = minimize(
                lambda x: _soft_min_obj_and_grad(x, beta_final)[0],
                x0,
                jac=lambda x: _soft_min_obj_and_grad(x, beta_final)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'ftol': 1e-14,
                    'gtol': 1e-10
                }
            )
            new_pts = res.x.reshape((N, 2))
            if np.all(np.isfinite(new_pts)):
                r_new = _compute_radius(new_pts)
                if r_new > best_r:
                    best_r = r_new
                    best_pts = new_pts
        except Exception:
            pass
    
    return best_pts

# Keep the function name exactly as requested
# The task says: "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.
