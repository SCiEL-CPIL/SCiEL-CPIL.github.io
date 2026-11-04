import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial.distance import pdist

N = 100
CUBE_HALF = 0.5

def compute_radius(centers):
    """
    Compute the actual maximum radius for a given set of centers.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Clip to ensure validity in case of floating point drift
    centers = np.clip(centers, -CUBE_HALF, CUBE_HALF)
    
    # Pairwise distances
    # Using pdist for efficiency and robustness
    dists = pdist(centers, metric='euclidean')
    min_pair_dist = np.min(dists) / 2.0
    
    # Distance to walls
    # dist_to_wall_i = min(CUBE_HALF - |c_ik|)
    dist_to_walls = CUBE_HALF - np.abs(centers)
    min_wall_dist = np.min(dist_to_walls)
    
    return min(min_pair_dist, min_wall_dist)

def penalty_objective_and_grad(centers_flat, R_target):
    """
    Objective: Sum of squared violations of radius constraint R_target.
    Violations:
    1. Pairwise: d_ij < 2*R_target
    2. Wall: w_i < R_target
    
    Returns: (objective_value, gradient)
    """
    centers = centers_flat.reshape(N, 3)
    
    # --- Pairwise Distances ---
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid div by 0 in gradient calc, mask later
    
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 0.0)
    
    # Violations: v_ij = max(0, 2*R_target - dists)
    # We want to minimize sum(v_ij^2)
    
    target_pair = 2.0 * R_target
    violations_pair = np.maximum(0.0, target_pair - dists)
    # Mask diagonal
    np.fill_diagonal(violations_pair, 0.0)
    
    obj_pair = np.sum(violations_pair**2)
    
    # Gradient calculation
    # grad_i = sum_j 2 * v_ij * (-(c_i - c_j) / d_ij)
    # Note: d_ij is symmetric.
    
    safe_dists = np.where(dists < 1e-8, 1.0, dists)
    # Unit vectors from i to j
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    np.fill_diagonal(unit_vecs, 0.0) # Zero out self-interaction
    
    # Weight by violation
    # grad_i = -2 * sum_j (v_ij / d_ij) * (c_i - c_j)
    # = -2 * sum_j (v_ij * unit_vec_ij)
    weights = violations_pair / safe_dists
    np.fill_diagonal(weights, 0.0)
    
    grad_pair = -2.0 * np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Distances ---
    # w_ik = CUBE_HALF - |c_ik|
    # Violation: v_ik = max(0, R_target - w_ik)
    # = max(0, R_target - (CUBE_HALF - |c_ik|))
    # = max(0, R_target - CUBE_HALF + |c_ik|)
    
    abs_centers = np.abs(centers)
    wall_dists = CUBE_HALF - abs_centers
    violations_wall = np.maximum(0.0, R_target - wall_dists)
    
    obj_wall = np.sum(violations_wall**2)
    
    # Gradient:
    # d(v^2)/d c_ik = 2 * v * d(R - w)/d c_ik
    # d(R - (CUBE_HALF - |c|))/d c = d(|c|)/d c = sign(c)
    # So grad_ik = 2 * v_ik * sign(c_ik)
    
    signs = np.sign(centers)
    # If c is 0, sign is 0. If R > 0.5 and c=0, v>0, grad=0. This is a saddle/flat spot.
    # Usually we start away from 0 or R is small.
    
    grad_wall = 2.0 * violations_wall * signs
    
    # Total
    obj = obj_pair + obj_wall
    grad = (grad_pair + grad_wall).flatten()
    
    return obj, grad

def soft_min_objective_and_grad(centers_flat, alpha=50.0):
    """
    Objective: -soft_min(distances)
    soft_min(d) = -1/alpha * log(sum(exp(-alpha * d)))
    Maximizing soft_min is equivalent to minimizing -soft_min.
    This is a smooth approximation to maximizing the minimum distance.
    """
    centers = centers_flat.reshape(N, 3)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0) # Avoid exp(-inf) or log(0) issues, mask later
    
    # Wall
    abs_centers = np.abs(centers)
    wall_dists = CUBE_HALF - abs_centers
    
    # Combine all distances into one vector for soft-min
    # We can do them separately or combine. Combining is cleaner for the log-sum-exp.
    # d_pair is N x N (with 0 on diag)
    # d_wall is N x 3
    
    # To avoid underflow in exp(-alpha * d), we can subtract min_d.
    # But for simplicity, if alpha is not too large, it's fine.
    
    # Mask diagonal of pairwise
    dists_masked = dists.copy()
    np.fill_diagonal(dists_masked, np.inf)
    
    all_dists = np.concatenate([dists_masked.flatten(), wall_dists.flatten()])
    
    # Soft min
    # s = -1/alpha * log(sum(exp(-alpha * d)))
    # To maximize s, minimize -s = 1/alpha * log(sum(exp(-alpha * d)))
    
    # Gradient of -s w.r.t d_k:
    # d(-s)/d d_k = 1/alpha * 1/sum(exp(-alpha d)) * (-alpha exp(-alpha d_k))
    # = - exp(-alpha d_k) / sum(exp(-alpha d))
    # Let p_k = exp(-alpha d_k) / sum(exp(-alpha d))
    # Then grad_d = -p_k
    
    # We need grad w.r.t centers.
    # d d_k / d c_i
    
    # This is complex to vectorize directly on the combined vector.
    # Let's compute gradients for pairwise and wall separately.
    
    # Pairwise Gradient
    # d_ij = ||c_i - c_j||
    # d d_ij / d c_i = (c_i - c_j) / d_ij
    # grad_i = sum_j p_ij * (c_i - c_j) / d_ij
    
    # Wall Gradient
    # w_ik = 0.5 - |c_ik|
    # d w_ik / d c_ik = -sign(c_ik)
    # grad_ik = p_ik * (-sign(c_ik))
    
    # Compute probabilities
    exp_pair = np.exp(-alpha * dists_masked)
    sum_exp_pair = np.sum(exp_pair)
    p_pair = exp_pair / sum_exp_pair
    
    exp_wall = np.exp(-alpha * wall_dists)
    sum_exp_wall = np.sum(exp_wall)
    p_wall = exp_wall / sum_exp_wall
    
    # Objective value
    obj = (1.0 / alpha) * (np.log(sum_exp_pair) + np.log(sum_exp_wall))
    
    # Gradients
    # Pairwise
    safe_dists = np.where(dists_masked < 1e-8, 1.0, dists_masked)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    np.fill_diagonal(unit_vecs, 0.0)
    
    grad_pair = np.sum(p_pair[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall
    signs = np.sign(centers)
    grad_wall = -p_wall * signs
    
    grad = (grad_pair + grad_wall).flatten()
    
    return obj, grad

def simulated_annealing(centers, max_iter=2000, T_start=0.05, T_end=0.001, alpha_decay=0.9995, deadline=None):
    """
    Multi-restart simulated annealing with adaptive step size.
    Uses pdist for fast distance calculation.
    """
    current_centers = centers.copy()
    current_radius = compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    T = T_start
    
    # Precompute indices for pdist? pdist is fast enough.
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Adaptive step size
        step_size = T * 0.5
        # Random perturbation
        noise = np.random.normal(0, step_size, current_centers.shape)
        new_centers = current_centers + noise
        new_centers = np.clip(new_centers, -CUBE_HALF, CUBE_HALF)
        
        new_radius = compute_radius(new_centers)
        
        delta_r = new_radius - current_radius
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / T):
            current_centers = new_centers
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
                
        T *= alpha_decay
        
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial configurations
    candidates = []
    
    # 1. Grid 5x5x4
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # 2. Grid 4x5x5
    m1, m2, m3 = 4, 5, 5
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # 3. Random (multiple seeds)
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        candidates.append(np.random.uniform(-0.45, 0.45, (N, 3)))
        
    # 4. Jittered Grid
    base_grid = candidates[0].copy()
    np.random.seed(123)
    jitter = np.random.normal(0, 0.01, (N, 3))
    candidates.append(np.clip(base_grid + jitter, -0.5, 0.5))
    
    # 5. FCC-like lattice (from inspiration)
    # Simple cubic is too sparse, let's try a denser starting point
    # 5x5x4 grid is actually quite good for 100 points.
    # Let's add a jittered FCC structure if possible, or just more randoms.
    for seed in [555, 777]:
        np.random.seed(seed)
        candidates.append(np.random.uniform(-0.48, 0.48, (N, 3)))

    for init_idx, init_centers in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
            
        current_centers = init_centers.copy()
        
        # Phase 1: Simulated Annealing for exploration
        # This helps escape local minima that gradient descent might get stuck in
        sa_deadline = min(deadline - 15.0, start_time + 10.0 + init_idx * 2.0)
        sa_centers, sa_radius = simulated_annealing(current_centers, max_iter=1500, deadline=sa_deadline)
        current_centers = sa_centers
        r_sa = sa_radius
        
        # Phase 2: Gradient Descent Polish (Penalty Method)
        # Use the best configuration found so far for this restart
        # We want to maximize the actual radius.
        
        # Start with a target radius slightly below the current best
        # to ensure convergence, then increase.
        R_target = max(0.05, r_sa * 0.95)
        max_R = 0.15
        
        # We will try to increase R_target in small steps
        # If we can't satisfy the constraint, we stop.
        
        while R_target < max_R and time.time() < deadline - 10.0:
            if time.time() > deadline - 10.0:
                break
                
            # Minimize penalty for current R_target
            # Use L-BFGS-B with bounds for better constraint handling
            try:
                bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
                res = opt.minimize(
                    penalty_objective_and_grad,
                    current_centers.flatten(),
                    args=(R_target,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'gtol': 1e-6, 'maxiter': 500, 'ftol': 1e-12}
                )
            except Exception:
                break
                
            current_centers = res.x.reshape(N, 3)
            current_centers = np.clip(current_centers, -CUBE_HALF, CUBE_HALF)
            
            # Check if constraint is satisfied
            final_obj, _ = penalty_objective_and_grad(current_centers.flatten(), R_target)
            
            if final_obj < 1e-4:
                # Successfully packed at R_target
                R_target += 0.001
            else:
                # Failed to pack at R_target, try a slightly smaller increment or break
                break
                
        # Phase 3: Soft-Min Polish
        # Use the best configuration found so far for this restart
        # We want to maximize the actual radius.
        # Minimize -soft_min(distances)
        
        if time.time() < deadline - 5.0:
            try:
                res_polish = opt.minimize(
                    soft_min_objective_and_grad,
                    current_centers.flatten(),
                    args=(50.0,), # alpha
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-CUBE_HALF, CUBE_HALF)] * (N * 3),
                    options={'gtol': 1e-8, 'maxiter': 300, 'ftol': 1e-12}
                )
                polished_centers = res_polish.x.reshape(N, 3)
                polished_centers = np.clip(polished_centers, -CUBE_HALF, CUBE_HALF)
                
                r_polished = compute_radius(polished_centers)
                
                # Also check the pre-polish radius
                r_pre = compute_radius(current_centers)
                
                if r_polished > r_pre:
                    current_centers = polished_centers
                    r_current = r_polished
                else:
                    r_current = r_pre
                    
            except Exception:
                r_current = compute_radius(current_centers)
        else:
            r_current = compute_radius(current_centers)
            
        if r_current > best_radius:
            best_radius = r_current
            best_centers = current_centers.copy()

    # Fallback
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
