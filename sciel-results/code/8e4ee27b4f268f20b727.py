"""
Strategy: Smooth Penalty Optimization + Active Set Constraint Refinement.
Instead of discrete moves or force physics, we treat the problem as a continuous
optimization of a smooth surrogate for the minimum distance, then refine by
solving the system of equations where the most binding constraints are active.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
        
    # Vectorized distance calculation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _logsumexp_min_penalty(centers, beta):
    """
    Smooth approximation of -min(distances).
    We want to maximize min(d_ij, d_wall).
    Minimizing the log-sum-exp of (1/d) is equivalent to maximizing the min d in a smooth way,
    but log-sum-exp of (1/d) is tricky near 0.
    
    Better approach: Maximize the "soft minimum".
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    This approximates min(x) as beta -> inf.
    We want to maximize soft_min(distances).
    So we minimize -soft_min(distances).
    """
    # Distances to walls
    # d_wall_i = min(0.5 - x_i, 0.5 - y_i, 0.5 + x_i, 0.5 + y_i)
    # It's easier to compute all 4*N wall distances and N(N-1)/2 pair distances.
    
    x = centers[:, 0]
    y = centers[:, 1]
    
    wall_dists = np.concatenate([
        0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y
    ])
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[~np.eye(N, dtype=bool)]
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Clip to avoid log(0) or overflow
    all_dists = np.maximum(all_dists, 1e-12)
    
    # Soft minimum: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow in exp(-beta * d) if beta is large and d is small,
    # we can subtract the min first, but we want the min itself.
    # Actually, for large beta, exp(-beta * d_min) dominates.
    # log(sum(exp(-beta * d))) approx -beta * d_min + log(count_of_min).
    # So -1/beta * log(...) approx d_min - (log(count))/beta.
    
    # Stable logsumexp computation
    max_term = np.max(-beta * all_dists) # This is -beta * min_d
    # Wait, exp(-beta * d). If d is small, -beta * d is large (less negative).
    # So max_term is the largest exponent.
    
    log_sum = max_term + np.log(np.sum(np.exp(-beta * all_dists - max_term)))
    soft_min = -1.0 / beta * log_sum
    
    return -soft_min # We want to minimize this (maximize soft_min)

def _objective_and_grad(centers, beta):
    """Objective and gradient for L-BFGS-B."""
    x = centers.flatten()
    
    # Objective
    val = _logsumexp_min_penalty(centers, beta)
    
    # Gradient (analytical)
    # f = -1/beta * log(sum(exp(-beta * d_i)))
    # df/dx = -1/beta * 1/S * sum( (-beta * exp(-beta * d_i)) * dd_i/dx )
    # df/dx = 1/S * sum( exp(-beta * d_i) * dd_i/dx )
    # where S = sum(exp(-beta * d_i))
    # Note: d_i are distances.
    
    x_c = centers[:, 0]
    y_c = centers[:, 1]
    
    wall_dists = np.concatenate([
        0.5 - x_c, 0.5 + x_c, 0.5 - y_c, 0.5 + y_c
    ])
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[~np.eye(N, dtype=bool)]
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    all_dists = np.maximum(all_dists, 1e-12)
    
    exp_terms = np.exp(-beta * all_dists)
    S = np.sum(exp_terms)
    
    # Weights w_i = exp(-beta * d_i) / S
    w = exp_terms / S
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # d(0.5 - x_i)/dx_i = -1, d(0.5 + x_i)/dx_i = 1, etc.
    # Indices 0..N-1: 0.5 - x_i -> grad_x -= w
    # Indices N..2N-1: 0.5 + x_i -> grad_x += w
    # Indices 2N..3N-1: 0.5 - y_i -> grad_y -= w
    # Indices 3N..4N-1: 0.5 + y_i -> grad_y += w
    
    grad[0::2] -= w[0:N]    # x minus
    grad[0::2] += w[N:2*N]  # x plus
    grad[1::2] -= w[2*N:3*N]# y minus
    grad[1::2] += w[3*N:4*N]# y plus
    
    # Pair gradients
    # d_ij = ||c_i - c_j||
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij = -(c_i - c_j) / d_ij
    
    # We need to map pair_dists back to the matrix
    # Let's just recompute the matrix for clarity, it's fast enough for N=50
    # dists matrix is (N, N)
    
    # w_pair is the subset of w corresponding to pairs
    w_pair = w[4*N:]
    
    # Reshape w_pair to (N, N) and zero out diagonal
    # The order of pair_dists from dists[~eye] is row-major, skipping diagonal.
    # It's easier to work with the full matrix.
    
    # Create full weight matrix
    W_full = np.zeros((N, N))
    # Fill W_full with w_pair in the same order
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                W_full[i, j] = w_pair[idx]
                idx += 1
    
    # Gradient contribution from pairs
    # sum_j w_ij * (c_i - c_j) / d_ij
    # This is equivalent to: for each i, sum_j (W_full[i,j] / dists[i,j]) * (c_i - c_j)
    
    # To avoid division by zero, dists are clipped in all_dists, but here we use raw dists.
    # We should use the same clipped dists for consistency in gradient direction?
    # If dist is 0, direction is undefined. But we clip in penalty.
    # Let's use a safe division.
    safe_dists = np.maximum(dists, 1e-12)
    
    # Coeff matrix C[i,j] = W_full[i,j] / safe_dists[i,j]
    C = W_full / safe_dists
    
    # grad_i = sum_j C[i,j] * (c_i - c_j)
    # = c_i * sum_j C[i,j] - sum_j C[i,j] * c_j
    
    sum_C = np.sum(C, axis=1)
    sum_C_c = C.dot(centers)
    
    grad_pair = centers * sum_C[:, np.newaxis] - sum_C_c
    
    grad += grad_pair
    
    return val, grad

def _optimize_smooth(centers, beta, max_iter=200, deadline=None):
    """Run L-BFGS-B optimization."""
    x0 = centers.flatten()
    
    def fun(x):
        if deadline and time.time() > deadline:
            raise Exception("Timeout")
        c = x.reshape(N, 2)
        val, _ = _objective_and_grad(c, beta)
        return val
        
    def jac(x):
        if deadline and time.time() > deadline:
            raise Exception("Timeout")
        c = x.reshape(N, 2)
        _, g = _objective_and_grad(c, beta)
        return g
        
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _active_set_refinement(centers, max_iter=100, deadline=None):
    """
    Refine by identifying the set of constraints active at the optimum.
    At the optimum, r = min(dists).
    The constraints d_ij - 2r >= 0 and d_wall - r >= 0 are active for some pairs.
    We can set up a system:
    For active pairs (i,j): ||c_i - c_j|| = 2r
    For active walls (i,k): dist(c_i, wall_k) = r
    
    This is a system of non-linear equations. We can solve for c and r.
    Number of variables: 2N + 1.
    Number of equations: M (number of active constraints).
    If M < 2N+1, the system is underdetermined. We need to maximize r subject to these?
    Actually, if we fix the active set, the optimal position is often determined by the geometry.
    If the active set forms a rigid structure, r is fixed.
    
    A simpler approach: Use the current r. Identify constraints where dist < r + epsilon.
    Force them to be exactly r.
    Then optimize the remaining free variables to maximize the minimum distance of the non-active constraints?
    This is getting complex.
    
    Alternative: Just use the smooth optimization with very high beta.
    """
    return centers

def construct():
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Generate diverse initializations
    inits = []
    
    # Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    inits.append(grid)
    
    # Hexagonal
    centers_hex = []
    for cols in range(5, 11):
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            continue
        sx = 1.0 / cols
        sy = 1.0 / rows
        for row in range(rows):
            for col in range(cols):
                if len(centers_hex) >= N:
                    break
                x = (col + 0.5) * sx - 0.5
                if row % 2 == 1:
                    x += sx / 2
                y = (row + 0.5) * sy - 0.5
                centers_hex.append([x, y])
        if len(centers_hex) >= N:
            break
    if len(centers_hex) == N:
        inits.append(np.array(centers_hex))
        
    # Random
    rng = np.random.RandomState(SEED)
    for i in range(5):
        inits.append(rng.uniform(-0.4, 0.4, size=(N, 2)))
        
    # 2. Optimize each initialization with increasing beta
    betas = [10, 50, 100, 200, 500]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        c = init.copy()
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
            
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Check remaining time
            rem = deadline - time.time()
            if rem < 2.0:
                break
                
            # Allocate time per optimization
            # L-BFGS-B is fast, but let's cap it
            max_iter = 50 if beta < 100 else 20
            
            try:
                c_opt = _optimize_smooth(c, beta, max_iter=max_iter, deadline=deadline)
                r_opt = _compute_radius(c_opt)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = c_opt.copy()
                c = c_opt
            except Exception:
                pass
                
    # 3. Final polish with very high beta on the best solution
    if best_centers is not None and time.time() < deadline - 1.0:
        # Try a few high beta steps
        for beta in [500, 1000, 2000]:
            if time.time() > deadline - 0.5:
                break
            try:
                c_polish = _optimize_smooth(best_centers, beta, max_iter=50, deadline=deadline)
                r_polish = _compute_radius(c_polish)
                if r_polish > best_radius:
                    best_radius = r_polish
                    best_centers = c_polish.copy()
            except Exception:
                break

    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers

# EVOLVE-BLOCK-END
