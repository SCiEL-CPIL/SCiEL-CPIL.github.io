import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(cents):
    """Compute the actual packing radius."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _soft_min_objective(params, n, beta):
    """
    Soft-minimum surrogate for the packing radius.
    We minimize the negative of the soft-min of all constraints.
    Constraints:
    1. Pairwise: |ci - cj| - 2r >= 0
    2. Wall: 0.5 - |cix| - r >= 0, etc.
    
    Actually, let's define the objective as minimizing the violation.
    Or better, maximize the soft-min of the "slack" variables.
    
    Let's use the standard soft-min:
    f(x) = -1/beta * log(sum(exp(-beta * g_i(x))))
    where g_i are the constraint functions (slack).
    We want to maximize f(x). So we minimize -f(x).
    
    However, computing exp for 1225+200 constraints might be slow or overflow.
    We can use a stable soft-min.
    
    Alternative: Minimize sum of squared violations for negative slack?
    No, L-BFGS-B works best on smooth functions.
    
    Let's define the objective as:
    Minimize - (soft_min of slacks)
    
    Slacks:
    s_ij = |ci - cj| - 2r
    s_iw = 0.5 - |ciw| - r
    
    We want all s >= 0.
    If we maximize the minimum s, we maximize r.
    
    Let's try minimizing the sum of squared negative parts (hinge loss) plus a term for r?
    No, the binary search approach is better.
    
    In the binary search approach:
    Fix r. Minimize the violation: sum(max(0, 2r - |ci-cj|)^2 + max(0, r + |ciw| - 0.5)^2).
    If min violation < tol, r is feasible.
    
    This objective is not smooth at 0. We can smooth it.
    Smooth hinge: max(0, x)^2 -> x^2 * sigmoid(x * alpha) or similar.
    Or just use L-BFGS-B on the sum of squares of the violations, which is smooth if we don't use max, but that penalizes non-violations too.
    
    Actually, a common trick is to minimize:
    sum( (2r - d_ij)^2 * I(d_ij < 2r) )
    This is non-differentiable.
    
    Let's use the soft-min approach directly on the radius.
    Objective: Minimize - soft_min(slacks, beta)
    Gradient: Can be computed.
    
    But L-BFGS-B requires bounds.
    Bounds for x, y: [-0.5, 0.5].
    Bounds for r: [0, 0.5].
    
    Let's implement the soft-min objective and gradient.
    """
    cents = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    # Compute slacks
    # Pairwise
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_slacks = dists - 2*r
    
    # Wall
    wall_slacks = []
    wall_slacks.append(0.5 - cents[:, 0] - r) # Left
    wall_slacks.append(cents[:, 0] + 0.5 - r) # Right
    wall_slacks.append(0.5 - cents[:, 1] - r) # Bottom
    wall_slacks.append(cents[:, 1] + 0.5 - r) # Top
    
    all_slacks = np.concatenate([pair_slacks.flatten(), np.array(wall_slacks).flatten()])
    
    # Soft min: -1/beta * log(sum(exp(-beta * s)))
    # Stable version:
    # Let m = min(s)
    # soft_min = m - 1/beta * log(sum(exp(-beta * (s - m))))
    
    m = np.min(all_slacks)
    # If m is very negative, exp(-beta*(s-m)) might overflow if s-m is large positive?
    # No, s-m >= 0, so exp(-beta*(s-m)) <= 1. No overflow.
    # Underflow is fine.
    
    exp_terms = np.exp(-beta * (all_slacks - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    
    obj = -soft_min_val
    
    # Gradient
    # d(soft_min)/ds_k = exp(-beta*(s_k - m)) / sum(exp(-beta*(s_k - m))) = p_k
    # p_k is the probability distribution.
    # d(soft_min)/dx = sum_k p_k * (d s_k / dx)
    
    p = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(params)
    
    # Pairwise slacks gradients
    # s_ij = d_ij - 2r
    # d s_ij / d ci = (ci - cj) / d_ij
    # d s_ij / d cj = (cj - ci) / d_ij
    # d s_ij / dr = -2
    
    # We need to accumulate gradients.
    # p is flattened. Index mapping:
    # Pairs: i < j. Index in all_slacks:
    # Let's iterate.
    
    pair_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pk = p[pair_idx]
            if pk > 1e-15:
                d = dists[i, j]
                if d > 1e-15:
                    diff_ij = cents[i] - cents[j]
                    grad[2*i:2*i+2] += pk * (diff_ij / d)
                    grad[2*j:2*j+2] += pk * (-diff_ij / d)
                    grad[2*n] += pk * (-2)
            pair_idx += 1
            
    # Wall slacks
    # Wall 0 (Left): s = 0.5 - x - r
    # Wall 1 (Right): s = x + 0.5 - r
    # Wall 2 (Bottom): s = 0.5 - y - r
    # Wall 3 (Top): s = y + 0.5 - r
    
    # Indices in all_slacks:
    # Pairs count: n*(n-1)/2
    w_start = pair_idx
    for i in range(n):
        pk_l = p[w_start + i]
        if pk_l > 1e-15:
            grad[2*i] += pk_l * (-1)
            grad[2*n] += pk_l * (-1)
            
        pk_r = p[w_start + n + i]
        if pk_r > 1e-15:
            grad[2*i] += pk_r * (1)
            grad[2*n] += pk_r * (-1)
            
        pk_b = p[w_start + 2*n + i]
        if pk_b > 1e-15:
            grad[2*i+1] += pk_b * (-1)
            grad[2*n] += pk_b * (-1)
            
        pk_t = p[w_start + 3*n + i]
        if pk_t > 1e-15:
            grad[2*i+1] += pk_t * (1)
            grad[2*n] += pk_t * (-1)
            
    return obj, grad

def _initial_grid(n):
    """Generate a grid initial placement."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _simulated_annealing(cents, time_limit=10.0, beta=50.0):
    """
    Perform simulated annealing on the soft-min objective.
    """
    start_time = time.time()
    n = len(cents)
    T = 0.05
    T_min = 1e-6
    alpha = 0.995
    
    best_cents = cents.copy()
    best_obj = _soft_min_objective(np.concatenate([cents.flatten(), [_compute_radius(cents)]]), n, beta)[0]
    # Note: The objective function uses r as a variable. In SA, we should probably fix r?
    # Or let r float?
    # If we let r float, the optimizer will push r up.
    # But the soft-min objective is tricky.
    
    # Let's stick to the binary search + L-BFGS-B strategy as it's more robust for "feasibility".
    # SA on the soft-min might get stuck.
    
    # Actually, the directive says: "Model the packing as a constrained non-linear least-squares problem using a differentiable soft-minimum surrogate... optimized via L-BFGS-B".
    # So I should implement L-BFGS-B on the soft-min.
    
    # Let's just return cents for now and implement the L-BFGS-B part in construct.
    return cents

def _lbfgs_optimize(cents, time_limit=10.0, beta=50.0):
    """
    Optimize using L-BFGS-B on the soft-min objective.
    """
    start_time = time.time()
    n = len(cents)
    
    r0 = _compute_radius(cents) * 0.99
    x0 = np.concatenate([cents.flatten(), [r0]])
    
    bounds = [(-0.5, 0.5)] * (2*n) + [(0, 0.5)]
    
    def fun_and_grad(p):
        obj, grad = _soft_min_objective(p, n, beta)
        return obj, grad
        
    try:
        res = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False}
        )
        
        if res.success or res.fun < fun_and_grad(x0)[0]:
            new_cents = res.x[:2*n].reshape(n, 2)
            new_r = _compute_radius(new_cents)
            if new_r > r0:
                return new_cents, new_r
    except Exception:
        pass
        
    return cents, r0

def _binary_search_radius(cents, r_low, r_high, time_limit=10.0):
    """
    Binary search for the maximum feasible radius.
    Feasibility check: Minimize violation using L-BFGS-B.
    """
    start_time = time.time()
    n = len(cents)
    
    def _violation_loss(p, r_target):
        cents = p[:2*n].reshape(n, 2)
        # Pairwise violations
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_viol = np.maximum(0, 2*r_target - dists)
        
        # Wall violations
        wall_viol_x_l = np.maximum(0, r_target + cents[:, 0] - 0.5)
        wall_viol_x_r = np.maximum(0, r_target - cents[:, 0] - 0.5)
        wall_viol_y_b = np.maximum(0, r_target + cents[:, 1] - 0.5)
        wall_viol_y_t = np.maximum(0, r_target - cents[:, 1] - 0.5)
        
        total_viol = np.sum(pair_viol**2) + np.sum(wall_viol_x_l**2) + np.sum(wall_viol_x_r**2) + np.sum(wall_viol_y_b**2) + np.sum(wall_viol_y_t**2)
        return total_viol
        
    # This loss is not differentiable at 0.
    # We can smooth it: max(0, x)^2 -> x^2 * sigmoid(x * k)
    # Or just use the hard max and let L-BFGS-B struggle? No, L-BFGS-B needs gradients.
    # Let's use a smooth approximation:
    # f(x) = 0.5 * (x + sqrt(x^2 + eps))
    # This is smooth.
    
    def _smooth_violation_loss(p, r_target, eps=1e-6):
        cents = p[:2*n].reshape(n, 2)
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # s = 2r - d
        s_pair = 2*r_target - dists
        # Smooth max: 0.5 * (s + sqrt(s^2 + eps))
        smooth_pair = 0.5 * (s_pair + np.sqrt(s_pair**2 + eps))
        
        s_wl = r_target + cents[:, 0] - 0.5
        s_wr = r_target - cents[:, 0] - 0.5
        s_wb = r_target + cents[:, 1] - 0.5
        s_wt = r_target - cents[:, 1] - 0.5
        
        smooth_wl = 0.5 * (s_wl + np.sqrt(s_wl**2 + eps))
        smooth_wr = 0.5 * (s_wr + np.sqrt(s_wr**2 + eps))
        smooth_wb = 0.5 * (s_wb + np.sqrt(s_wb**2 + eps))
        smooth_wt = 0.5 * (s_wt + np.sqrt(s_wt**2 + eps))
        
        loss = np.sum(smooth_pair**2) + np.sum(smooth_wl**2) + np.sum(smooth_wr**2) + np.sum(smooth_wb**2) + np.sum(smooth_wt**2)
        return loss
        
    # Gradient of smooth max:
    # d/dx [0.5(x + sqrt(x^2+eps))] = 0.5(1 + x/sqrt(x^2+eps))
    
    def _smooth_violation_grad(p, r_target, eps=1e-6):
        cents = p[:2*n].reshape(n, 2)
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        grad = np.zeros_like(p)
        
        s_pair = 2*r_target - dists
        # d smooth_max / d s = 0.5(1 + s/sqrt(s^2+eps))
        ds_pair = 0.5 * (1 + s_pair / np.sqrt(s_pair**2 + eps))
        # d loss / d s_pair = 2 * smooth_pair * ds_pair
        d_loss_ds_pair = 2 * (0.5 * (s_pair + np.sqrt(s_pair**2 + eps))) * ds_pair
        
        # d s_pair / d ci = - (ci - cj) / dists
        # d s_pair / d cj = (ci - cj) / dists
        
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d > 1e-15:
                    diff_ij = cents[i] - cents[j]
                    coeff = d_loss_ds_pair[i, j]
                    grad[2*i:2*i+2] -= coeff * (diff_ij / d)
                    grad[2*j:2*j+2] += coeff * (diff_ij / d)
                    
        # Walls
        s_wl = r_target + cents[:, 0] - 0.5
        s_wr = r_target - cents[:, 0] - 0.5
        s_wb = r_target + cents[:, 1] - 0.5
        s_wt = r_target - cents[:, 1] - 0.5
        
        ds_wl = 0.5 * (1 + s_wl / np.sqrt(s_wl**2 + eps))
        ds_wr = 0.5 * (1 + s_wr / np.sqrt(s_wr**2 + eps))
        ds_wb = 0.5 * (1 + s_wb / np.sqrt(s_wb**2 + eps))
        ds_wt = 0.5 * (1 + s_wt / np.sqrt(s_wt**2 + eps))
        
        d_loss_ds_wl = 2 * (0.5 * (s_wl + np.sqrt(s_wl**2 + eps))) * ds_wl
        d_loss_ds_wr = 2 * (0.5 * (s_wr + np.sqrt(s_wr**2 + eps))) * ds_wr
        d_loss_ds_wb = 2 * (0.5 * (s_wb + np.sqrt(s_wb**2 + eps))) * ds_wb
        d_loss_ds_wt = 2 * (0.5 * (s_wt + np.sqrt(s_wt**2 + eps))) * ds_wt
        
        grad[2*range(n)] += d_loss_ds_wl
        grad[2*range(n)] -= d_loss_ds_wr
        grad[2*range(n)+1] += d_loss_ds_wb
        grad[2*range(n)+1] -= d_loss_ds_wt
        
        return grad
        
    def _is_feasible(r_target, start_cents, time_left):
        x0 = np.concatenate([start_cents.flatten(), [0.0]]) # r not used in loss, but keep shape consistent? No, loss only uses positions.
        # Wait, the loss function I defined only takes positions.
        # Let's just optimize positions.
        x0 = start_cents.flatten()
        
        bounds = [(-0.5, 0.5)] * (2*n)
        
        def fun(p):
            return _smooth_violation_loss(p, r_target)
            
        def grad(p):
            return _smooth_violation_grad(p, r_target)
            
        try:
            res = minimize(
                fun,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            if time.time() - start_time > time_left:
                return False, start_cents
            final_loss = _smooth_violation_loss(res.x, r_target)
            if final_loss < 1e-8:
                return True, res.x.reshape(n, 2)
            else:
                return False, res.x.reshape(n, 2)
        except Exception:
            return False, start_cents

    # Binary search
    r_low = _compute_radius(cents)
    r_high = 0.15 # Upper bound guess
    
    best_cents = cents.copy()
    
    while (r_high - r_low) > 1e-6 and time.time() - start_time < time_limit:
        r_mid = (r_low + r_high) / 2
        feasible, new_cents = _is_feasible(r_mid, best_cents, time_limit - (time.time() - start_time))
        if feasible:
            r_low = r_mid
            best_cents = new_cents
        else:
            r_high = r_mid
            # Use new_cents as next start? It might be closer to feasible.
            best_cents = new_cents
            
    return best_cents, r_low

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # 1. Initial Grid
    cents = _initial_grid(N)
    best_cents = cents.copy()
    best_r = _compute_radius(cents)
    
    # 2. L-BFGS-B Optimization on Soft-Min
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        opt_cents, opt_r = _lbfgs_optimize(best_cents, time_limit=min(remaining - 5, 20), beta=50)
        if opt_r > best_r:
            best_r = opt_r
            best_cents = opt_cents
            
    # 3. Binary Search with Feasibility Checks
    remaining = deadline - (time.time() - start_time)
    if remaining > 20:
        bs_cents, bs_r = _binary_search_radius(best_cents, best_r, 0.15, time_limit=min(remaining - 5, 60))
        if bs_r > best_r:
            best_r = bs_r
            best_cents = bs_cents
            
    # 4. Final Polish with L-BFGS-B
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        opt_cents, opt_r = _lbfgs_optimize(best_cents, time_limit=min(remaining - 2, 10), beta=100)
        if opt_r > best_r:
            best_r = opt_r
            best_cents = opt_cents
            
    return best_cents

# EVOLVE-BLOCK-END
