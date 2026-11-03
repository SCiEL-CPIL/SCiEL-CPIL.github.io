"""
Packing 100 equal spheres in a unit cube using soft-min surrogate optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return float(r)

def _soft_min_objective_and_grad(x, alpha):
    """
    Objective function: minimize -soft_min(distances)
    Which is equivalent to maximizing soft_min(distances).
    
    d_m(x) are the distances to constraints:
    1. Wall distances: 0.5 - |x_i| for each coordinate
    2. Pairwise distances: ||x_i - x_j|| / 2
    
    soft_min(d) = -1/alpha * log( (1/M) * sum(exp(-alpha * d_m)) )
    
    We minimize f(x) = -soft_min(d) = 1/alpha * log( (1/M) * sum(exp(-alpha * d_m)) )
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Compute all constraint distances
    # 1. Wall distances
    # For each sphere i, distance to nearest wall is min(0.5 - x_i, 0.5 + x_i, ...)
    # Actually, distance to wall is 0.5 - max(|x_i|, |y_i|, |z_i|)
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    # 2. Pairwise distances
    # diff[i, j] = x_i - x_j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # We only care about half the pairwise distance
    pair_dists = dists / 2.0
    
    # Collect all distances into a single list for soft-min
    # Wall dists: N values
    # Pair dists: N*(N-1)/2 values
    # Total M = N + N*(N-1)/2
    
    # For efficiency, we compute the log-sum-exp directly
    # f(x) = 1/alpha * log( (1/M) * ( sum_wall(exp(-alpha * w_i)) + sum_pair(exp(-alpha * p_ij)) ) )
    
    # To avoid underflow/overflow, we use the log-sum-exp trick
    # Let Z = sum(exp(-alpha * d_m))
    # log Z = max_d(-alpha * d_m) + log(sum(exp(-alpha * d_m - max_d(-alpha * d_m))))
    
    # We'll compute the terms separately
    
    # Wall terms
    # w_i = 0.5 - max_abs_i
    # exp_term_w_i = exp(-alpha * w_i)
    # Note: if w_i < 0, exp(-alpha * w_i) is large. But we clip centers to box, so w_i >= 0.
    
    # Pair terms
    # p_ij = dists[i,j] / 2
    # exp_term_p_ij = exp(-alpha * p_ij)
    
    # We only need the upper triangle for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    p_active = pair_dists[i_idx, j_idx]
    
    # Calculate log-sum-exp for walls
    # w_active = wall_dists
    # log_sum_exp_w = -alpha * max(w) + log(sum(exp(-alpha * (w - max(w)))))
    # But simpler: just compute the sum if alpha is not too large.
    # For alpha up to ~100, exp(-100 * 0.1) is ~1e-5, safe.
    # exp(-100 * 0.01) is ~1e-5.
    # If d is small, exp is close to 1.
    
    # Let's use a stable formulation.
    # log_sum_exp(A) = max(A) + log(sum(exp(A - max(A))))
    
    # A_w = -alpha * wall_dists
    A_w = -alpha * wall_dists
    max_w = np.max(A_w)
    log_sum_w = max_w + np.log(np.sum(np.exp(A_w - max_w)))
    
    # A_p = -alpha * p_active
    A_p = -alpha * p_active
    max_p = np.max(A_p)
    log_sum_p = max_p + np.log(np.sum(np.exp(A_p - max_p)))
    
    # Combine
    # log(Z) = log(exp(log_sum_w) + exp(log_sum_p))
    # log(Z) = max(log_sum_w, log_sum_p) + log(exp(log_sum_w - max) + exp(log_sum_p - max))
    max_log_sum = np.max([log_sum_w, log_sum_p])
    log_Z = max_log_sum + np.log(np.exp(log_sum_w - max_log_sum) + np.exp(log_sum_p - max_log_sum))
    
    # M = N + N*(N-1)/2
    M = n + n*(n-1)/2.0
    
    # f(x) = 1/alpha * (log_Z - log(M))
    val = (log_Z - np.log(M)) / alpha
    
    # Gradient
    # f(x) = 1/alpha * log( (1/M) * sum(exp(-alpha * d_m)) )
    # df/dx = 1/alpha * ( 1 / sum(exp(-alpha * d_m)) * sum( d(exp(-alpha * d_m))/dx ) )
    # d(exp(-alpha * d_m))/dx = exp(-alpha * d_m) * (-alpha * dd_m/dx)
    # df/dx = 1/alpha * ( 1 / Z * sum( exp(-alpha * d_m) * (-alpha * dd_m/dx) ) )
    # df/dx = -1 / Z * sum( exp(-alpha * d_m) * dd_m/dx )
    # Note: We are minimizing f(x) = -soft_min. 
    # soft_min = -1/alpha * log(Z/M). 
    # d(soft_min)/dx = -1/alpha * (1/Z * sum(exp(-alpha d) * (-alpha dd/dx))) = 1/Z * sum(exp(-alpha d) * dd/dx)
    # So d(-soft_min)/dx = -1/Z * sum(exp(-alpha d) * dd/dx)
    
    # Let's compute the weights
    # w_m = exp(-alpha * d_m) / Z
    # grad = - sum( w_m * dd_m/dx )
    
    # We need to compute the weighted sum of gradients of distances.
    # grad_wall: for each sphere i, d(w_i)/dx_i
    # w_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    # dd/dx is non-zero only for the coordinate with max abs value.
    # If x_i > 0, d/dx_i = -1. If x_i < 0, d/dx_i = 1.
    
    grad = np.zeros_like(centers)
    
    # Wall gradients
    # weights_w = exp(-alpha * wall_dists)
    # We need to normalize by Z.
    # Let's compute unnormalized weights first, then normalize.
    
    # Actually, it's easier to compute the gradient of the log-sum-exp directly.
    # grad log_Z = sum( (exp(A_m) / Z) * dA_m/dx )
    # A_m = -alpha * d_m
    # dA_m/dx = -alpha * dd_m/dx
    
    # So grad f = 1/alpha * grad log_Z = 1/alpha * sum( (exp(A_m)/Z) * (-alpha * dd_m/dx) )
    # grad f = - sum( (exp(A_m)/Z) * dd_m/dx )
    
    # Let's compute the sum of exp(A_m) * dd_m/dx for walls and pairs separately.
    
    # Walls
    # A_w_i = -alpha * w_i
    # exp_A_w = np.exp(A_w)
    # We need sum_i exp_A_w_i * d(w_i)/dx_i
    
    # d(w_i)/dx_i is a vector.
    # For sphere i, let k_i be the index of max abs coord.
    # d(w_i)/dx_i[k_i] = -sign(x_i[k_i])
    # Others are 0.
    
    max_abs_idx = np.argmax(abs_centers, axis=1)
    signs = np.sign(centers[np.arange(n), max_abs_idx])
    # If sign is 0, treat as 1
    signs[signs == 0] = 1.0
    
    # d_w_dx is a (N, 3) array, mostly zeros
    d_w_dx = np.zeros_like(centers)
    d_w_dx[np.arange(n), max_abs_idx] = -signs
    
    # Weighted sum for walls
    # S_w = sum_i exp(A_w_i) * d_w_dx[i]
    # exp_A_w = np.exp(A_w - max_w) * np.exp(max_w)
    # We can compute relative weights to avoid overflow, then scale.
    # But we already have log_Z.
    # Let's just compute exp(A_w) safely.
    # If alpha is large, A_w can be very negative.
    
    # To be safe, let's compute the gradient using the probabilities p_m = exp(A_m)/Z
    # p_w_i = exp(A_w_i) / Z
    # p_p_ij = exp(A_p_ij) / Z
    
    # We can compute log_p_w_i = A_w_i - log_Z
    # p_w_i = exp(A_w_i - log_Z)
    
    log_p_w = A_w - log_Z
    p_w = np.exp(log_p_w)
    
    # S_w = sum_i p_w_i * d_w_dx[i]
    S_w = np.sum(p_w[:, np.newaxis] * d_w_dx, axis=0)
    
    # Pairs
    # A_p_ij = -alpha * p_ij
    # log_p_p = A_p - log_Z
    log_p_p = A_p - log_Z
    p_p = np.exp(log_p_p)
    
    # d(p_ij)/dx_i = (x_i - x_j) / ||x_i - x_j||
    # d(p_ij)/dx_j = (x_j - x_i) / ||x_i - x_j||
    
    # We have i_idx, j_idx for active pairs
    # diff_active = centers[i_idx] - centers[j_idx]
    # dist_active = dists[i_idx, j_idx]
    # unit_vec = diff_active / dist_active
    
    diff_active = centers[i_idx] - centers[j_idx]
    dist_active = dists[i_idx, j_idx]
    
    # Avoid division by zero (should not happen due to +1e-15)
    unit_vec = diff_active / dist_active[:, np.newaxis]
    
    # d_p_dx_i = unit_vec
    # d_p_dx_j = -unit_vec
    
    # S_p = sum_ij p_p_ij * ( d_p_dx_i + d_p_dx_j )
    # Note: the sum is over all pairs.
    # We accumulate into grad.
    
    # S_p_i = sum_{j != i} p_p_ij * unit_vec_ij
    # S_p_j = sum_{i != j} p_p_ij * (-unit_vec_ij)
    
    # Using np.add.at
    S_p = np.zeros_like(centers)
    np.add.at(S_p, i_idx, p_p[:, np.newaxis] * unit_vec)
    np.add.at(S_p, j_idx, p_p[:, np.newaxis] * (-unit_vec))
    
    # Total gradient
    grad = -(S_w + S_p)
    
    return val, grad.flatten()

def _initialize_bcc():
    """
    Initialize centers using a BCC lattice structure.
    BCC is a good packing for spheres.
    We need 100 spheres.
    A 5x4x5 grid of BCC unit cells?
    Each BCC unit cell has 2 spheres (corners + center).
    But corners are shared.
    Let's just use a simple grid and add BCC-like offsets.
    Or generate a dense grid and pick the best 100.
    
    Alternative: Use a 5x4x5 grid of points, and for each grid point, add a small offset.
    Better: Generate a BCC lattice that fits in the cube.
    Side length L=1.
    BCC lattice constant a.
    Number of atoms in BCC with side L: (L/a)^3 * 2.
    If we want 100 atoms, (L/a)^3 * 2 = 100 => (L/a)^3 = 50 => L/a = 50^(1/3) ~ 3.68.
    So a ~ 1/3.68 ~ 0.27.
    
    Let's generate a BCC lattice with a=0.27.
    x = i*a + offset
    y = j*a + offset
    z = k*a + offset
    Plus body center: (i+0.5)*a, (j+0.5)*a, (k+0.5)*a
    
    We will generate a slightly larger set and clip/keep those inside.
    """
    a = 0.27
    nx = int(np.ceil(1.0 / a)) + 2
    points = []
    
    for i in range(-1, nx + 1):
        for j in range(-1, nx + 1):
            for k in range(-1, nx + 1):
                # Corner
                p1 = np.array([i*a, j*a, k*a])
                # Body center
                p2 = np.array([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
                
                if np.all(np.abs(p1) <= 0.5):
                    points.append(p1)
                if np.all(np.abs(p2) <= 0.5):
                    points.append(p2)
    
    points = np.array(points)
    
    # If we have more than N, pick N random ones or the ones with highest min-dist
    if len(points) > N:
        # Calculate min-dist for each point to others? Too slow.
        # Just pick random N
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Pad with random points
        extra = np.random.uniform(-0.5, 0.5, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    
    # Add small noise
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, size=(N, 3))
    points = np.clip(points, -0.49, 0.49)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # 1. Initialization
    best_pos = _initialize_bcc()
    best_r = _compute_radius(best_pos)
    
    # 2. Soft-Min Annealing Optimization
    # We want to maximize min_dist.
    # We minimize f(x, alpha) = -soft_min(dist, alpha).
    # As alpha -> infinity, f(x, alpha) -> -min(dist).
    
    # Start with small alpha, increase over time.
    alpha_start = 5.0
    alpha_end = 100.0
    
    # Number of optimization steps
    num_steps = 50
    
    current_alpha = alpha_start
    pos = best_pos.copy()
    
    for step in range(num_steps):
        if time.time() > deadline - 20.0:
            break
            
        # Linear annealing of alpha
        progress = step / num_steps
        current_alpha = alpha_start * (1 - progress) + alpha_end * progress
        
        # Optimize for current alpha
        # Use L-BFGS-B
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                pos.flatten(),
                args=(current_alpha,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            new_pos = res.x.reshape(N, 3)
            
            # Check if improved
            new_r = _compute_radius(new_pos)
            if new_r > best_r:
                best_r = new_r
                best_pos = new_pos
            pos = new_pos
        except Exception:
            pass
            
    # 3. Final Polish with High Alpha
    # Use a very high alpha to sharpen the minimum
    current_alpha = 200.0
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                best_pos.flatten(),
                args=(current_alpha,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            new_pos = res.x.reshape(N, 3)
            new_r = _compute_radius(new_pos)
            if new_r > best_r:
                best_r = new_r
                best_pos = new_pos
        except Exception:
            pass
            
    # Final validation and clipping
    best_pos = np.clip(best_pos, -0.499999, 0.499999)
    
    return best_pos
