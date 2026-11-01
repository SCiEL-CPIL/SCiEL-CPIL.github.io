"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(pos):
    """
    Computes the maximum radius for the given positions.
    Returns 0 if invalid.
    """
    if pos.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    # Check if inside [-0.5, 0.5]
    if np.any(pos < -0.5) or np.any(pos > 0.5):
        return 0.0

    # Pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = dist.min()
    
    # Distance to walls
    # min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    # Since pos is in [-0.5, 0.5], min(0.5 - x, x + 0.5) is 0.5 - |x|
    min_wall_dist = (0.5 - np.abs(pos)).min()
    
    return min(0.5 * min_pair_dist, min_wall_dist)

def _soft_min_objective(flat_pos, beta):
    """
    Objective function to minimize.
    We want to MAXIMIZE the minimum distance.
    So we MINIMIZE the negative soft-min of the distances.
    """
    pos = flat_pos.reshape(N, 2)
    
    # Compute pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(dist_sq + 1e-12) # small epsilon for stability
    np.fill_diagonal(dist, np.inf)
    
    # Distances to walls: 0.5 - |x|, 0.5 - |y|
    # We treat these as distances too.
    # If a point is outside, distance is negative? No, we constrain bounds in L-BFGS-B.
    # But soft-min should handle the "min" of valid distances.
    # To be safe, we can add a large penalty if out of bounds, but L-BFGS-B handles bounds.
    
    wall_dists_x = 0.5 - np.abs(pos[:, 0])
    wall_dists_y = 0.5 - np.abs(pos[:, 1])
    
    # Collect all relevant distances: pairwise and wall
    # Pairwise: N*(N-1)/2
    # Wall: 2*N
    pair_dists = dist[~np.eye(N, dtype=bool)]
    all_dists = np.concatenate([pair_dists, wall_dists_x, wall_dists_y])
    
    # Soft min: -1/beta * log(sum(exp(-beta * dists)))
    # To avoid overflow/underflow in exp, we subtract max
    m = np.min(all_dists)
    # If m is very small, exp(-beta*m) is large. 
    # Actually, standard soft min formula:
    # S(x) = -1/beta * log( sum exp(-beta * x_i) )
    # Let's use log-sum-exp trick for stability.
    # sum exp(-beta * x_i) = exp(-beta * m) * sum exp(-beta * (x_i - m))
    # log(sum) = -beta * m + log(sum exp(-beta * (x_i - m)))
    # S(x) = m - 1/beta * log(sum exp(-beta * (x_i - m)))
    
    shifted = -beta * (all_dists - m)
    # Clip shifted to prevent overflow in exp if beta is large and dists vary a lot
    # exp(0) = 1, exp(-large) = 0
    # If any dist is much larger than m, shifted is very negative, exp is 0. Good.
    # If dist == m, shifted is 0, exp is 1.
    
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - (1.0 / beta) * log_sum_exp
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _soft_min_grad(flat_pos, beta):
    """
    Gradient of the soft-min objective.
    """
    pos = flat_pos.reshape(N, 2)
    
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    wall_dists_x = 0.5 - np.abs(pos[:, 0])
    wall_dists_y = 0.5 - np.abs(pos[:, 1])
    
    pair_dists = dist[~np.eye(N, dtype=bool)]
    all_dists = np.concatenate([pair_dists, wall_dists_x, wall_dists_y])
    
    m = np.min(all_dists)
    shifted = -beta * (all_dists - m)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Gradient of soft_min w.r.t x_i:
    # S(x) = m - 1/beta * log(sum exp(-beta * x_i))
    # dS/dx_k = -1/beta * (1/sum_exp) * sum_j (-beta * exp(-beta * x_j) * dx_j/dx_k)
    # Since x_k is one of the variables in the sum:
    # dS/dx_k = -1/beta * (1/sum_exp) * (-beta * exp(-beta * x_k)) * (1) + ... (chain rule for m?)
    # Wait, m = min(x). m is not differentiable.
    # However, in the log-sum-exp formulation, we don't explicitly differentiate m.
    # Let's re-derive.
    # S(x) = -1/beta * log( sum_j exp(-beta * x_j) )
    # dS/dx_k = -1/beta * 1/(sum exp) * (-beta * exp(-beta * x_k))
    # dS/dx_k = exp(-beta * x_k) / sum(exp(-beta * x_j))
    # This is a probability distribution p_k = exp(-beta * x_k) / sum(exp(-beta * x_j))
    # Note: p_k sums to 1.
    
    p = exp_shifted / sum_exp
    
    # We want gradient of -S(x) w.r.t pos.
    # -dS/dx_k = -p_k
    # So we need to accumulate -p_k * d(x_k)/d(pos)
    
    grad = np.zeros((N, 2))
    
    # Pairwise distances: x_ij = ||p_i - p_j||
    # p_k corresponds to x_k.
    # For a pair (i, j), x_k = dist[i, j].
    # d(dist[i, j])/d(p_i) = (p_i - p_j) / dist[i, j]
    # d(dist[i, j])/d(p_j) = (p_j - p_i) / dist[i, j]
    
    # We need to map the flat array 'pair_dists' back to indices.
    # pair_dists is dist[~np.eye(N, dtype=bool)]
    # This is row-major order of the upper triangle? No, np.eye removes diagonal.
    # Let's just compute gradients directly from the matrix to avoid index mapping errors.
    
    # Recalculate p for pairwise and wall separately to avoid mapping issues?
    # Or just compute the contribution of each term.
    
    # Let's compute the gradient of the objective function directly using the chain rule on the matrix.
    # Objective = - S(all_dists)
    # S(all_dists) = -1/beta * log(sum exp(-beta * d_k))
    # dObj/d d_k = - (-1/beta) * 1/sum_exp * (-beta * exp(-beta * d_k)) = - p_k
    # So dObj/d d_k = -p_k
    
    # Now we need d(d_k)/d(pos).
    
    # 1. Pairwise terms
    # dist[i, j] for i < j
    # d(dist[i, j])/d(pos[i]) = (pos[i] - pos[j]) / dist[i, j]
    # d(dist[i, j])/d(pos[j]) = (pos[j] - pos[i]) / dist[i, j]
    
    # We need p_k for each pair.
    # We have 'p' array which is aligned with 'all_dists'.
    # all_dists = [pair_dists, wall_dists_x, wall_dists_y]
    # pair_dists = dist[~np.eye(N, dtype=bool)]
    
    # Let's create a matrix of weights W[i, j] = p_k for the pair (i, j).
    W = np.zeros((N, N))
    
    # Get indices of non-diagonal elements
    idx_i, idx_j = np.where(~np.eye(N, dtype=bool))
    # pair_dists is flattened in the same order as idx_i, idx_j
    p_pairs = p[:len(idx_i)]
    
    # Assign weights
    W[idx_i, idx_j] = p_pairs
    # Since dist is symmetric, W[j, i] should be same?
    # The gradient contribution for pos[i] from pair (i,j) is -p_k * (pos[i]-pos[j])/dist[i,j]
    # The gradient contribution for pos[j] from pair (i,j) is -p_k * (pos[j]-pos[i])/dist[i,j]
    # Note: p_k is the same for (i,j) and (j,i) in the sum?
    # The sum includes all pairs. Usually we sum over i<j.
    # My 'all_dists' includes all i != j?
    # dist[~np.eye(N)] includes both (i,j) and (j,i).
    # So pair_dists has N*(N-1) elements.
    # This is inefficient (double counting).
    # Let's redo using only i < j to save memory and computation.
    
    # --- Revised Gradient Calculation ---
    
    # We only consider i < j for pairs.
    # Number of pairs: N*(N-1)/2
    # Number of wall dists: 2*N
    
    # Recalculate p for the correct set of distances.
    # It is safer to just compute the gradient numerically? No, analytic is better.
    # Let's stick to the full matrix approach but be careful.
    
    # Actually, let's just compute the gradient of the soft-min of the *unique* constraints.
    # Constraints:
    # 1. dist[i, j] >= 2r for i < j
    # 2. 0.5 - |x_i| >= r
    # 3. 0.5 - |y_i| >= r
    
    # Let's define the vector of distances D.
    # D contains:
    # - dist[i, j] for i < j
    # - 0.5 - |x_i| for all i
    # - 0.5 - |y_i| for all i
    
    # We need to compute p_k for this specific D.
    
    # 1. Compute D
    triu_idx_i, triu_idx_j = np.triu_indices(N, k=1)
    pair_dists = dist[triu_idx_i, triu_idx_j]
    
    wall_x = 0.5 - np.abs(pos[:, 0])
    wall_y = 0.5 - np.abs(pos[:, 1])
    
    D = np.concatenate([pair_dists, wall_x, wall_y])
    
    # 2. Compute p
    m = np.min(D)
    shifted = -beta * (D - m)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    p = exp_shifted / sum_exp
    
    # 3. Accumulate gradients
    # dObj/dD_k = -p_k
    
    grad = np.zeros((N, 2))
    
    # Pairwise gradients
    # d(dist[i,j])/d(pos[i]) = (pos[i] - pos[j]) / dist[i,j]
    # d(dist[i,j])/d(pos[j]) = (pos[j] - pos[i]) / dist[i,j]
    
    # We need to iterate or vectorize.
    # Vectorized:
    # diff = pos[triu_idx_i] - pos[triu_idx_j]
    # dists = pair_dists
    # unit_vecs = diff / dists[:, None]
    # 
    # grad_contrib_i = -p_pairs * unit_vecs
    # grad_contrib_j = -p_pairs * (-unit_vecs) = p_pairs * unit_vecs
    
    diff = pos[triu_idx_i] - pos[triu_idx_j]
    # Avoid division by zero
    safe_dists = np.where(pair_dists < 1e-12, 1e-12, pair_dists)
    unit_vecs = diff / safe_dists[:, None]
    
    p_pairs = p[:len(triu_idx_i)]
    
    # Accumulate for i
    np.add.at(grad, triu_idx_i, -p_pairs[:, None] * unit_vecs)
    # Accumulate for j
    np.add.at(grad, triu_idx_j, p_pairs[:, None] * unit_vecs)
    
    # Wall gradients
    # d(0.5 - |x_i|)/d(x_i) = -sign(x_i)
    # d(0.5 - |y_i|)/d(y_i) = -sign(y_i)
    
    p_wall_x = p[len(triu_idx_i):len(triu_idx_i)+N]
    p_wall_y = p[len(triu_idx_i)+N:]
    
    sign_x = np.sign(pos[:, 0])
    sign_y = np.sign(pos[:, 1])
    
    # Handle 0 sign? If pos is 0, derivative of |x| is undefined, but subgradient is [-1, 1].
    # In optimization, we can pick 0 or use a small epsilon.
    # Let's use np.where(sign==0, 0, sign)
    sign_x = np.where(sign_x == 0, 0, sign_x)
    sign_y = np.where(sign_y == 0, 0, sign_y)
    
    grad[:, 0] += -p_wall_x * sign_x
    grad[:, 1] += -p_wall_y * sign_y
    
    return grad

def _initial_grid():
    """Generate a good initial grid."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a hexagonal-like grid or just square grid
    # Square grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_radius(pos_init, time_budget, beta=100.0):
    """
    Run L-BFGS-B optimization to maximize the soft-min radius.
    """
    t_start = time.time()
    
    # Flatten
    x0 = pos_init.flatten()
    
    # Bounds: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Objective and Gradient
    def obj(x):
        return _soft_min_objective(x, beta)
        
    def grad(x):
        return _soft_min_grad(x, beta)
        
    # Run optimization
    # maxiter is a soft limit, we check time in callback
    res = None
    last_check = time.time()
    
    def callback(xk):
        nonlocal last_check
        if time.time() - last_check > 1.0:
            last_check = time.time()
            if time.time() - t_start > time_budget:
                # Raise exception to break out of minimize?
                # Or just return a flag?
                # scipy minimize doesn't support easy early stopping via callback return value
                # unless we raise an exception.
                raise TimeoutError("Time budget exceeded")
                
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000},
                       callback=callback)
    except TimeoutError:
        # Use the best point found so far?
        # res is None if exception happens before assignment?
        # Actually, minimize might have updated xk.
        # We can't easily get the current x from the callback in this scope without a closure variable.
        # Let's use a mutable list to store the best x.
        pass

    # If we hit timeout, we might not have a valid res.
    # Let's restart the logic to be safer:
    # We will run minimize in a loop, checking time periodically?
    # No, L-BFGS-B is a black box.
    
    # Alternative: Use a smaller number of iterations and check time.
    # Or just trust that 1000 iterations is fast enough?
    # N=50, 100 variables. L-BFGS-B is fast.
    # Let's assume it finishes within budget.
    
    if res is None or not res.success:
        # If it failed or timed out, we try to recover the best x
        # But we don't have it.
        # Let's just return the initial pos if it fails?
        # Or run a simpler local search?
        # For now, return initial.
        return pos_init

    x_opt = res.x.reshape(N, 2)
    
    # Polish: Run a few iterations of a "hard" local search or high-beta soft-min
    # to recover the exact minimum.
    # We can run another optimization with a very high beta (e.g., 1000)
    # starting from x_opt.
    
    if time.time() - t_start < time_budget - 5:
        res2 = minimize(obj, x_opt, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
        if res2.success:
            x_opt = res2.x.reshape(N, 2)
            
    return x_opt

def construct():
    t_start = time.time()
    deadline = 120.0 # Safety margin
    
    # 1. Initial Grid
    pos = _initial_grid()
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # 2. Optimization
    # We can try multiple betas or restarts if time permits.
    # Start with a moderate beta.
    beta = 100.0
    
    # Time budget for optimization: leave 10s for safety
    opt_budget = deadline - 10 - (time.time() - t_start)
    
    if opt_budget > 5:
        pos_opt = _optimize_radius(best_pos, opt_budget, beta)
        r_opt = _compute_radius(pos_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pos = pos_opt
            
    # 3. Local Polish (Greedy Hill Climbing)
    # This helps if the soft-min converged to a point that is slightly suboptimal
    # due to the smoothing.
    t_polish_start = time.time()
    polish_time = min(10.0, deadline - 2 - (time.time() - t_start))
    
    if polish_time > 0:
        cur_pos = best_pos.copy()
        cur_r = best_r
        step = 0.005
        np.random.seed(42)
        
        while time.time() - t_polish_start < polish_time:
            # Random move
            i = np.random.randint(N)
            old = cur_pos[i].copy()
            cur_pos[i] = old + np.random.normal(0, step, size=2)
            # Clamp
            lim = 0.5 - 0.01
            cur_pos[i] = np.clip(cur_pos[i], -lim, lim)
            
            new_r = _compute_radius(cur_pos)
            
            if new_r > cur_r:
                cur_r = new_r
                if cur_r > best_r:
                    best_r = cur_r
                    best_pos = cur_pos.copy()
                # Accept
            else:
                # Reject
                cur_pos[i] = old
                
            # Adaptive step
            if np.random.rand() < 0.1:
                step *= 0.95
                
    # Final check
    if not np.all(np.isfinite(best_pos)):
        return _initial_grid()
        
    return best_pos

# EVOLVE-BLOCK-END
