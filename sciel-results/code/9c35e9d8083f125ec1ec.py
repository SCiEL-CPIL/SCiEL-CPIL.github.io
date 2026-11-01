"""
Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Strategy:
1. Initialize with hexagonal patterns.
2. Coarse optimization with L-BFGS-B on smooth repulsive energy.
3. Multi-scale soft-min surrogate optimization (annealing alpha) using L-BFGS-B.
4. Final subgradient polish for fine-tuning.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf

    return min(min_pair_dist / 2.0, min_boundary)

def _energy_and_grad(params, power=4.0):
    """
    Compute smooth repulsive energy and its gradient for coarse phase.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)

    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    d2_neg2 = 1.0 / (d2_safe**2)
    d2_neg3 = 1.0 / (d2_safe**3)
    np.fill_diagonal(d2_neg2, 0.0)
    np.fill_diagonal(d2_neg3, 0.0)

    E_pair = np.triu(d2_neg2, 1).sum()

    grad_x = -4.0 * (dx * d2_neg3).sum(axis=1)
    grad_y = -4.0 * (dy * d2_neg3).sum(axis=1)

    eps = 1e-8
    dx_w = np.clip(0.5 - x, eps, None)
    dx_w_neg = np.clip(0.5 + x, eps, None)
    dy_w = np.clip(0.5 - y, eps, None)
    dy_w_neg = np.clip(0.5 + y, eps, None)

    E_wall = (1.0/dx_w**2 + 1.0/dx_w_neg**2 + 1.0/dy_w**2 + 1.0/dy_w_neg**2).sum()

    grad_x += 2.0 / dx_w**3 - 2.0 / dx_w_neg**3
    grad_y += 2.0 / dy_w**3 - 2.0 / dy_w_neg**3

    total_E = E_pair + E_wall
    total_grad = np.concatenate([grad_x, grad_y])

    return total_E, total_grad

def _soft_min_surrogate(params, alpha):
    """
    Compute the soft-min surrogate objective and gradient.
    We want to maximize min(d_i). 
    Surrogate: S(x, alpha) = -1/alpha * log(sum(exp(-alpha * d_i)))
    Gradient: grad S = -1/sum(exp(-alpha*d_i)) * sum(exp(-alpha*d_i) * (-alpha * grad d_i))
             = sum(exp(-alpha*d_i) * grad d_i) / sum(exp(-alpha*d_i))
             
    Here d_i are the scaled distances: 
    For pairs: d_ij = ||c_i - c_j|| / 2
    For walls: d_iw = 0.5 - |c_ik|
    
    We compute the gradient of the distances with respect to c.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # 1. Compute distances and their gradients
    # Pairwise
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 1.0) # Avoid div by 0, though we mask later
    
    # Gradient of d_ij w.r.t c_i: (c_i - c_j) / d_ij
    # Gradient of d_ij w.r.t c_j: (c_j - c_i) / d_ij
    inv_d = 1.0 / d
    np.fill_diagonal(inv_d, 0.0)
    
    # grad_x_pair[i] = sum_j (x_i - x_j) / d_ij
    grad_x_pair = (dx * inv_d).sum(axis=1)
    grad_y_pair = (dy * inv_d).sum(axis=1)
    
    # Wall distances
    # d_left = x + 0.5, d_right = 0.5 - x
    # d_down = y + 0.5, d_up = 0.5 - y
    # We treat these as separate "distances" in the min set.
    
    d_left = x + 0.5
    d_right = 0.5 - x
    d_down = y + 0.5
    d_up = 0.5 - y
    
    # Gradients of wall distances
    # grad d_left / dx = 1
    # grad d_right / dx = -1
    # etc.
    
    # Collect all distances into a flat array for the log-sum-exp
    # Note: Pair distances are counted twice in the full matrix, but we only want unique pairs?
    # The min over all pairs is the same as min over unique pairs. 
    # Using the full matrix is fine for the sum, but doubles the weight of pair distances vs wall distances?
    # To be fair, we should probably use unique pairs.
    # Let's use upper triangle for pairs.
    
    triu_idx = np.triu_indices(N, k=1)
    pair_d = d[triu_idx]
    
    # Gradients for unique pairs
    # This is tricky to vectorize simply for the full gradient vector.
    # Alternative: Just use the full matrix but divide pair contribution by 2? 
    # Or just accept that pair constraints are "stronger" in the sum. 
    # Actually, standard soft-min uses all constraints. 
    # Let's stick to unique pairs to keep the count balanced with walls (4N walls vs ~2N^2 pairs).
    # The number of pairs is huge (19900), walls are 800.
    # The log-sum-exp will be dominated by the smallest distances.
    
    # Let's construct the gradient vectors for pairs carefully.
    # grad_x_pair_unique[i] = sum_{j>i} (x_i - x_j)/d_ij - sum_{j<i} (x_j - x_i)/d_ji 
    # This is equivalent to the full sum grad_x_pair computed above?
    # grad_x_pair_full[i] = sum_j (x_i - x_j)/d_ij.
    # Yes, the full sum is the correct gradient for the sum of distances.
    # If we use unique pairs in the objective, we must use half the gradient for pairs.
    # Let's just use the full matrix for simplicity, it's a valid surrogate (just different weighting).
    # Actually, using full matrix means each pair (i,j) contributes d_ij and d_ji (same value).
    # So the sum is 2 * sum_unique. The log-sum-exp will be slightly different but monotonic.
    
    # Let's use the full matrix for d, but we need to be careful with memory/time.
    # N=200, 40000 elements. Fine.
    
    # Combine all distances
    # We need to map the gradients back to the parameter vector.
    # The objective is f(c) = -1/alpha * log( sum_pairs exp(-alpha * d_ij) + sum_walls exp(-alpha * d_w) )
    
    # To avoid overflow/underflow in exp, use log-sum-exp trick.
    # But we also need the gradient, which involves the weights w_k = exp(-alpha * d_k) / sum(exp(...)).
    
    # Let's compute the weights.
    # d_flat = concat(pair_d_flat, wall_d_flat)
    # grad_flat = concat(pair_grad_flat, wall_grad_flat)
    
    # This is getting complex to vectorize perfectly without large temporary arrays.
    # Let's use a simpler approach: 
    # Compute the "effective" gradient as a weighted average of the individual distance gradients.
    
    # Weights for pairs:
    # w_ij = exp(-alpha * d_ij)
    # Sum of weights for pairs: S_pair = sum_{i,j} w_ij
    # Contribution to grad from pairs: sum_{i,j} w_ij * grad(d_ij)
    
    # Weights for walls:
    # w_w = exp(-alpha * d_w)
    # Sum of weights for walls: S_wall = sum_w w_w
    # Contribution to grad from walls: sum_w w_w * grad(d_w)
    
    # Total Sum S = S_pair + S_wall
    # Objective = -1/alpha * log(S)
    # Gradient = (1/S) * ( sum_pairs w_ij grad(d_ij) + sum_walls w_w grad(d_w) )
    #           = (1/S) * ( S_pair * E[grad_pair | w] + S_wall * E[grad_wall | w] )
    
    # Let's compute S_pair and the weighted gradient sum for pairs.
    # d is full matrix.
    # w = exp(-alpha * d)
    # We need to handle the diagonal (d=1.0, w=exp(-alpha)). We should exclude diagonal.
    mask_diag = np.eye(N, dtype=bool)
    w = np.exp(-alpha * d)
    w[mask_diag] = 0.0
    
    S_pair = w.sum()
    
    # Weighted gradient for pairs
    # grad_x_pair_weighted[i] = sum_j w_ij * (x_i - x_j) / d_ij
    # We already have grad_x_pair = sum_j (x_i - x_j) / d_ij
    # But we need weighted sum.
    # weighted_grad_x_pair = (w * inv_d * dx).sum(axis=1) ?
    # Note: grad(d_ij) w.r.t x_i is (x_i - x_j)/d_ij = dx_ij * inv_d_ij.
    # So yes.
    
    weighted_grad_x_pair = (w * inv_d * dx).sum(axis=1)
    weighted_grad_y_pair = (w * inv_d * dy).sum(axis=1)
    
    # Walls
    # d_left, d_right, d_down, d_up
    # We treat them as 4 separate constraints per circle.
    w_left = np.exp(-alpha * d_left)
    w_right = np.exp(-alpha * d_right)
    w_down = np.exp(-alpha * d_down)
    w_up = np.exp(-alpha * d_up)
    
    S_wall = w_left.sum() + w_right.sum() + w_down.sum() + w_up.sum()
    
    # Weighted gradients for walls
    # grad d_left / dx = 1
    # grad d_right / dx = -1
    # grad d_down / dy = 1
    # grad d_up / dy = -1
    
    weighted_grad_x_wall = w_left - w_right
    weighted_grad_y_wall = w_down - w_up
    
    S_total = S_pair + S_wall
    
    if S_total < 1e-300:
        # Fallback if underflow
        obj = 0.0
        grad = np.zeros(2*N)
        return obj, grad
        
    obj = -1.0 / alpha * np.log(S_total)
    
    # Total Gradient
    # grad_x = (weighted_grad_x_pair + weighted_grad_x_wall) / S_total
    # grad_y = (weighted_grad_y_pair + weighted_grad_y_wall) / S_total
    
    grad_x = (weighted_grad_x_pair + weighted_grad_x_wall) / S_total
    grad_y = (weighted_grad_y_pair + weighted_grad_y_wall) / S_total
    
    grad = np.concatenate([grad_x, grad_y])
    
    return obj, grad

def _optimize_soft_min(c_init, time_limit=30.0):
    """
    Multi-temperature soft-min optimization.
    """
    start_time = time.time()
    c = c_init.copy()
    
    # Alpha schedule: Start smooth (low alpha), end sharp (high alpha)
    # Low alpha -> smooth, wide basin. High alpha -> sharp, close to true min.
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit - 5:
            break
            
        def func_and_grad(params):
            if time.time() - start_time > time_limit:
                raise TimeoutError
            return _soft_min_surrogate(params, alpha)
        
        try:
            res = minimize(
                func_and_grad,
                c.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            if res.success or res.status == 0:
                c_new = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
                # Check if radius improved
                r_old = _compute_radius(c)
                r_new = _compute_radius(c_new)
                if r_new >= r_old - 1e-6: # Allow small decrease for convergence
                    c = c_new
        except Exception:
            pass
            
    return c

def _initialize_hexagonal(d_spacing=None):
    """Initialize with a hexagonal pattern with given spacing."""
    if d_spacing is None:
        d_spacing = 0.07

    points = []
    row_spacing = d_spacing * np.sqrt(3) / 2
    x_offset = d_spacing / 2

    y = -0.5 + d_spacing / 2
    row = 0
    while y <= 0.5 - d_spacing / 2:
        x = -0.5 + d_spacing / 2
        if row % 2 == 1:
            x += x_offset
        while x <= 0.5 - d_spacing / 2:
            points.append([x, y])
            x += d_spacing
        y += row_spacing
        row += 1

    points = np.array(points)
    if len(points) >= N:
        c = points[:N].copy()
        c = np.clip(c, -0.49, 0.49)
        return c
    return None

def _initialize_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _subgradient_polish(c, max_iters=2000, time_limit=30.0, base_step=1e-4):
    """
    Subgradient method on the non-smooth objective -min(d).
    Directly targets the closest pairs/walls.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    step = base_step
    stall_count = 0
    max_stall = 20

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        # Compute distances
        x = c[:, 0]
        y = c[:, 1]
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx**2 + dy**2
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        # Wall distances
        w_left = x + 0.5
        w_right = 0.5 - x
        w_down = y + 0.5
        w_up = 0.5 - y
        w = np.minimum(np.minimum(w_left, w_right), np.minimum(w_down, w_up))
        
        # Current radius
        min_pair_d = np.min(d)
        min_wall_d = np.min(w)
        current_r = min(min_pair_d / 2.0, min_wall_d)
        
        if current_r > best_r:
            best_r = current_r
            best_c = c.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                step *= 0.5
                if step < 1e-7:
                    break

        # Compute Subgradient
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        # Only consider pairs that are close to the minimum (active constraints)
        threshold_d = min_pair_d * 1.05
        mask_d = (d < threshold_d) & (d > 0)
        
        active_pairs = np.argwhere(mask_d)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        for i, j in active_pairs:
            di = d[i, j]
            if di < 1e-12: continue
            inv_d = 1.0 / di
            gx = (x[i] - x[j]) * inv_d
            gy = (y[i] - y[j]) * inv_d
            grad_x[i] += gx
            grad_y[i] += gy
            grad_x[j] -= gx
            grad_y[j] -= gy
            
        # Wall subgradients
        threshold_w = min_wall_d * 1.05
        mask_wl = (w_left < threshold_w)
        grad_x[mask_wl] += 1.0
        mask_wr = (w_right < threshold_w)
        grad_x[mask_wr] -= 1.0
        mask_wd = (w_down < threshold_w)
        grad_y[mask_wd] += 1.0
        mask_wu = (w_up < threshold_w)
        grad_y[mask_wu] -= 1.0
        
        # Normalize gradient
        g_norm = np.sqrt(grad_x**2 + grad_y**2)
        g_norm = np.where(g_norm > 1e-12, g_norm, 1.0)
        
        new_x = x + step * grad_x / g_norm
        new_y = y + step * grad_y / g_norm
        
        # Project to box
        new_x = np.clip(new_x, -0.499, 0.499)
        new_y = np.clip(new_y, -0.499, 0.499)
        
        c = np.column_stack((new_x, new_y))
        
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin

    # Generate initializations
    inits = []
    for d in np.linspace(0.065, 0.075, 5):
        c_hex = _initialize_hexagonal(d)
        if c_hex is not None:
            inits.append(c_hex)
    inits.append(_initialize_grid())
    
    # Pick best init
    init_results = []
    for c_init in inits:
        c_init = np.clip(c_init, -0.49, 0.49)
        r_init = _compute_radius(c_init)
        init_results.append((r_init, c_init))
        
    init_results.sort(key=lambda x: x[0], reverse=True)
    best_c = init_results[0][1].copy()
    best_r = init_results[0][0]

    # Phase 1: L-BFGS-B on smooth energy for coarse optimization
    if time.time() < deadline - 80:
        try:
            bounds = [(-0.499, 0.499)] * (2 * N)
            res = minimize(
                _energy_and_grad,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000, 'maxls': 50}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        except Exception:
            pass

    # Phase 2: Soft-Min Surrogate Optimization
    if time.time() < deadline - 20:
        time_for_soft = min(60.0, deadline - time.time() - 20)
        c_soft, r_soft = None, 0.0
        try:
            c_soft = _optimize_soft_min(best_c, time_limit=time_for_soft)
            r_soft = _compute_radius(c_soft)
            if r_soft > best_r:
                best_r = r_soft
                best_c = c_soft.copy()
        except Exception:
            pass

    # Phase 3: Subgradient polish for fine-tuning
    if time.time() < deadline - 5:
        time_for_polish = min(15.0, deadline - time.time() - 5)
        c_polished, r_polished = _subgradient_polish(best_c, max_iters=3000, time_limit=time_for_polish, base_step=2e-4)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    # Phase 4: Final tiny polish
    if time.time() < deadline - 1:
        time_for_final = min(3.0, deadline - time.time() - 1)
        c_final, r_final = _subgradient_polish(best_c, max_iters=1000, time_limit=time_for_final, base_step=5e-5)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()

    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
