import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the radius r for a given set of points.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    Returns 0.0 if invalid.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check bounds
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0

    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Compute distance to walls
    # Distance to x-walls: 0.5 - |x|
    # Distance to y-walls: 0.5 - |y|
    wall_dists = np.min(np.stack([0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min(values, beta):
    """
    Smooth approximation of min(values).
    values: array of shape (M,)
    beta: temperature parameter. Higher beta -> sharper min.
    """
    # To avoid overflow, subtract the max (or min depending on sign)
    # We are minimizing -soft_min, so we want soft_min to be close to min.
    # soft_min = -1/beta * log(sum(exp(-beta * v)))
    # To stabilize:
    v_min = np.min(values)
    # If v is very large, exp(-beta*v) is small.
    # If v is very small, exp(-beta*v) is large.
    # We subtract the minimum value to keep exponents <= 1? No, exp(-beta*(v - v_min)) <= 1.
    shifted = values - v_min
    exp_sum = np.sum(np.exp(-beta * shifted))
    return v_min - (1.0 / beta) * np.log(exp_sum)

def _objective_soft_min(points, beta):
    """
    Objective function for optimization: -soft_min(pair_dists/2, wall_dists)
    We want to maximize the min radius, so we minimize the negative.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Extract upper triangle for i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Wall distances
    wall_x = 0.5 - np.abs(x)
    wall_y = 0.5 - np.abs(y)
    wall_dists = np.minimum(wall_x, wall_y)
    
    # Combine all constraints into one vector
    all_vals = np.concatenate([pair_dists, wall_dists])
    
    # Calculate soft min
    r_soft = _soft_min(all_vals, beta)
    
    return -r_soft

def _objective_grad(points, beta):
    """
    Gradient of the objective function.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    wall_x = 0.5 - np.abs(x)
    wall_y = 0.5 - np.abs(y)
    wall_dists = np.minimum(wall_x, wall_y)
    
    all_vals = np.concatenate([pair_dists, wall_dists])
    
    # Soft min derivative
    # f = -1/beta * log(sum(exp(-beta * v)))
    # df/dv_k = (exp(-beta * v_k)) / sum(exp(-beta * v_j))
    
    shifted = all_vals - np.min(all_vals)
    exps = np.exp(-beta * shifted)
    sum_exps = np.sum(exps)
    weights = exps / sum_exps
    
    grad = np.zeros((N, 2))
    
    # Pairwise contributions
    # d(pair_dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # Note: pair_dists are halved, so d(pair_dist/2)/dx_i = 0.5 * (x_i - x_j) / dist_ij
    
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = weights[k]
        
        dist = dists[i, j]
        if dist < 1e-12:
            # Random perturbation if coincident
            grad[i, 0] += w * 0.5 * (np.random.rand() - 0.5)
            grad[i, 1] += w * 0.5 * (np.random.rand() - 0.5)
            grad[j, 0] -= w * 0.5 * (np.random.rand() - 0.5)
            grad[j, 1] -= w * 0.5 * (np.random.rand() - 0.5)
        else:
            # Derivative of 0.5 * dist
            # d(0.5 * sqrt((xi-xj)^2 + ...))/dx_i = 0.5 * (xi-xj)/dist
            factor = 0.5 / dist
            grad[i, 0] += w * factor * (x[i] - x[j])
            grad[i, 1] += w * factor * (y[i] - y[j])
            grad[j, 0] -= w * factor * (x[i] - x[j])
            grad[j, 1] -= w * factor * (y[i] - y[j])
            
    # Wall contributions
    # Wall dists are in the second half of all_vals
    wall_start = len(i_idx)
    for k in range(N):
        # x-wall
        wx_idx = wall_start + 2*k
        wy_idx = wall_start + 2*k + 1
        
        # Actually, wall_dists was min(wall_x, wall_y) per point?
        # In _objective_soft_min, I concatenated pair_dists and wall_dists.
        # wall_dists array has length N.
        # Wait, in _objective_soft_min:
        # wall_dists = np.minimum(wall_x, wall_y) -> shape (N,)
        # So there are N wall terms.
        # But in the gradient loop above, I assumed 2N wall terms?
        # Let's check _objective_soft_min again.
        # all_vals = concatenate([pair_dists, wall_dists])
        # len(pair_dists) = N*(N-1)/2
        # len(wall_dists) = N
        # So total length = N*(N-1)/2 + N
        
        # The loop above for walls is incorrect if I only have N wall terms.
        # Let's fix the gradient calculation for walls.
        pass

    # Correct Wall Gradient Calculation
    # wall_dists[k] = min(0.5 - |x_k|, 0.5 - |y_k|)
    # This is non-differentiable. The soft min smooths it, but we need the gradient of the soft min w.r.t x_k.
    # d(soft_min)/dx_k = sum_j (weights_j * d(val_j)/dx_k)
    
    # The wall terms in all_vals correspond to indices wall_start ... wall_start + N - 1
    # val_j for wall term k is min(0.5 - |x_k|, 0.5 - |y_k|)
    # This is tricky because of the min inside.
    # However, the soft min is over the set of values {pair_dists, wall_dists}.
    # If we treat wall_dists as separate inputs, we need d(wall_dist_k)/dx_k.
    
    # Let's simplify: Instead of analytic gradient for the complex wall min,
    # we can just use numerical gradient or a simpler objective.
    # Or, we can avoid the min in wall_dists by treating 4 walls separately?
    # r = min(min_pair/2, min_x_dist, min_y_dist)
    # If we expand wall_dists into 4N terms (left, right, bottom, top), the soft min handles it.
    # Let's refactor _objective_soft_min to use 4N wall terms.
    
    # For now, this gradient function is incomplete/incorrect for the current _objective_soft_min.
    # I will rewrite the polish step to use numerical gradient or a different method if analytic is too hard.
    # Or I can just use 'Nelder-Mead' which doesn't need gradient.
    # Nelder-Mead is robust but slower per iteration.
    # Given N=100 variables, Nelder-Mead might be okay for a short polish.
    
    return grad # Placeholder

def _force_directed_layout(points, num_iters=500, learning_rate=0.01):
    """
    Uses repulsive forces to spread points.
    """
    points = points.copy()
    for _ in range(num_iters):
        forces = np.zeros_like(points)
        
        # Repulsive forces between all pairs
        # F_ij = k * (r_ij / d_ij^3) * (p_i - p_j) ? 
        # Or simpler: F_ij = k * (p_i - p_j) / d_ij^2
        # We want to maximize min distance, so we push them apart.
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by 0
        
        # Inverse square law
        # Force magnitude proportional to 1/d^2
        # Direction is along the vector
        # F_i = sum_{j != i} (p_i - p_j) / d_ij^2
        
        # To vectorize:
        # inv_d2 = 1 / dists^2
        # inv_d2[diag] = 0
        # forces = sum over j of (p_i - p_j) * inv_d2[i,j]
        
        inv_d2 = 1.0 / (dists**2)
        np.fill_diagonal(inv_d2, 0.0)
        
        # F_i = sum_j (p_i - p_j) * inv_d2[i,j]
        # (p_i - p_j) is diff[i,j]
        # forces[i] = sum_j diff[i,j] * inv_d2[i,j]
        # This can be done with matrix multiplication?
        # diff is (N, N, 2). inv_d2 is (N, N).
        # We need to sum over j for each i and each dimension.
        
        # forces[:, 0] = np.sum(diff[:, :, 0] * inv_d2, axis=1)
        # forces[:, 1] = np.sum(diff[:, :, 1] * inv_d2, axis=1)
        
        forces[:, 0] = np.sum(diff[:, :, 0] * inv_d2, axis=1)
        forces[:, 1] = np.sum(diff[:, :, 1] * inv_d2, axis=1)
        
        # Attractive force to center to prevent drift?
        # Or just let repulsion handle it and clip.
        # If we only have repulsion, they will fly to corners.
        # We want them to fill the square.
        # A common trick is to add a weak attraction to the center or to the "ideal" grid positions.
        # Let's add a weak attraction to the center (0,0).
        forces -= 0.1 * points
        
        # Update
        points += learning_rate * forces
        
        # Clip to bounds
        points = np.clip(points, -0.5, 0.5)
        
    return points

def construct():
    start_time = time.time()
    deadline = 118.0  # 2s safety margin
    
    # 1. Initialization
    # Start with a jittered grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Add 50th point
    # Place it randomly or at a specific spot
    # A 7x7 grid has 49 points. The 50th is the problem.
    # Let's place the 50th point in the center (0,0) and let FDL move it.
    p50 = np.array([0.0, 0.0])
    points = np.vstack([grid, p50])
    
    # Jitter
    np.random.seed(42) # Reproducibility for debugging, but random is fine
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    best_points = points.copy()
    best_score = compute_radius(best_points)
    
    # 2. Force Directed Layout (FDL)
    # Run FDL for a short time to spread out the points and find a good local minimum.
    fdl_time_budget = 10.0
    fdl_start = time.time()
    
    # Run multiple short FDL runs with different initial jitters?
    # Or one long FDL run.
    # Let's do one FDL run with adaptive learning rate.
    
    lr = 0.05
    while time.time() - fdl_start < fdl_time_budget:
        # Run a batch of FDL iterations
        # Estimate how many iterations we can do
        # Let's just run 100 iterations per loop
        for _ in range(100):
            forces = np.zeros_like(points)
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            
            inv_d2 = 1.0 / (dists**2)
            np.fill_diagonal(inv_d2, 0.0)
            
            forces[:, 0] = np.sum(diff[:, :, 0] * inv_d2, axis=1)
            forces[:, 1] = np.sum(diff[:, :, 1] * inv_d2, axis=1)
            
            # Weak attraction to center
            forces -= 0.05 * points
            
            points += lr * forces
            points = np.clip(points, -0.5, 0.5)
            
        # Decay learning rate
        lr *= 0.95
        
        # Check score
        curr_score = compute_radius(points)
        if curr_score > best_score:
            best_score = curr_score
            best_points = points.copy()
            
    # 3. Simulated Annealing (SA)
    # Optimize the exact radius.
    sa_start = time.time()
    sa_budget = 80.0
    
    current_points = best_points.copy()
    current_score = best_score
    
    T = 0.01
    T_min = 1e-7
    alpha = 0.9995
    
    # Adaptive step size
    step_size = 0.02
    
    # We will run SA until time budget is exhausted
    while time.time() - sa_start < sa_budget:
        idx = np.random.randint(N)
        
        # Gaussian move
        move = np.random.normal(0, step_size, 2)
        new_points = current_points.copy()
        new_points[idx] += move
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = compute_radius(new_points)
        
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                # If we found a new best, maybe increase step size slightly?
                # Or keep it.
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat?
            # Let's just stop cooling and keep searching
            # Or break if we've been at T_min for a while?
            # For now, just clamp T.
            
        # Adaptive step size: if we haven't improved in a while, reduce step size
        # Simple heuristic: reduce step size slowly
        step_size *= 0.9999
        
    # 4. Polish with Gradient Descent / Soft Min
    # Use the best points found by SA.
    polish_start = time.time()
    polish_budget = deadline - (time.time() - start_time)
    
    if polish_budget > 2.0:
        # Try to polish using scipy.optimize.minimize
        # We need a smooth objective.
        # Let's use a soft-min objective.
        
        def obj_func(pts):
            return _objective_soft_min(pts, beta=100.0)
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        # Flatten points
        x0 = best_points.flatten()
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success or res.fun < obj_func(x0):
                new_pts = res.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                new_score = compute_radius(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts
        except Exception:
            pass
            
        # If we have more time, try a second polish with higher beta
        if time.time() - start_time < deadline - 2.0:
            def obj_func_high_beta(pts):
                return _objective_soft_min(pts, beta=500.0)
            
            try:
                res = minimize(obj_func_high_beta, best_points.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
                if res.success or res.fun < obj_func_high_beta(best_points.flatten()):
                    new_pts = res.x.reshape(N, 2)
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_score = compute_radius(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
            except Exception:
                pass

    return best_points
