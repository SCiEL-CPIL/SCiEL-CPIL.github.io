"""
Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Strategy:
1. Initialize with hexagonal patterns.
2. Coarse optimization with L-BFGS-B on smooth repulsive energy.
3. Hard-constrained NLP with SLSQP: maximize t subject to d_ij >= 2t and wall >= t.
4. Final subgradient polish for fine-tuning.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _slsqp_nlp(c_init, t_init, time_limit=30.0):
    """
    Hard-constrained NLP using SLSQP.
    Maximizes t subject to:
    1. ||c_i - c_j|| >= 2t  for all i < j
    2. c_ik >= -0.5 + t     for all i, k
    3. c_ik <=  0.5 - t     for all i, k
    
    We minimize -t.
    Variables: z = [c_1x, c_1y, ..., c_Nx, c_Ny, t]
    """
    start_time = time.time()
    
    def objective(z):
        if time.time() - start_time > time_limit:
            raise TimeoutError("SLSQP time limit exceeded")
        return -z[-1] # Maximize t

    def constraints(z):
        if time.time() - start_time > time_limit:
            raise TimeoutError("SLSQP time limit exceeded")
        
        c = z[:-1].reshape(N, 2)
        t = z[-1]
        
        cons = []
        
        # Pairwise distance constraints: ||c_i - c_j||^2 - (2t)^2 >= 0
        # SLSQP expects g(z) >= 0
        # Using squared distances to avoid sqrt, but note this changes the gradient shape slightly.
        # Actually, SLSQP handles non-linear constraints fine. 
        # d_ij^2 - 4t^2 >= 0
        # This is a valid inequality.
        
        # To speed up, we might only enforce constraints that are close to active? 
        # No, SLSQP needs all. 
        # However, 19900 constraints is a lot. 
        # Let's try to generate them.
        
        # Optimization: If we know t is small, most constraints are inactive. 
        # But SLSQP doesn't know that a priori.
        
        # For N=200, 19900 constraints might be too slow for SLSQP in 30s.
        # Alternative: Use a penalty method or just rely on L-BFGS-B + Subgradient.
        # The directive asks for SLSQP. Let's try a subset or a different formulation.
        
        # Actually, for N=200, SLSQP with 19900 constraints is likely to time out or fail.
        # A common trick is to only include constraints where distance < 2.5*t or similar, 
        # but this makes the problem non-static.
        
        # Let's try to implement it but with a check: if it's too slow, we return current best.
        # Given the risk, I will implement a "sparse" constraint generator that only includes 
        # pairs currently close to the limit. This is a heuristic but often works for local refinement.
        
        c_curr = c
        d2 = np.sum((c_curr[:, None, :] - c_curr[None, :, :])**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Only constrain pairs where d2 < (2.2 * t)^2
        # This keeps the constraint set small.
        threshold = (2.2 * t)**2
        mask = d2 < threshold
        idx_i, idx_j = np.where(mask)
        
        # Ensure i < j
        mask_ij = idx_i < idx_j
        idx_i = idx_i[mask_ij]
        idx_j = idx_j[mask_ij]
        
        for i, j in zip(idx_i, idx_j):
            ci = c[i]
            cj = c[j]
            # Constraint: (ci - cj)^2 - 4t^2 >= 0
            def con_func(z, i=i, j=j):
                if time.time() - start_time > time_limit:
                    raise TimeoutError("SLSQP time limit exceeded")
                ci = z[2*i:2*i+2]
                cj = z[2*j:2*j+2]
                t_val = z[-1]
                d2_val = np.sum((ci - cj)**2)
                return d2_val - 4.0 * t_val**2
            cons.append({'type': 'ineq', 'fun': con_func})
            
        # Boundary constraints:
        # c_ik + 0.5 - t >= 0  => c_ik >= t - 0.5
        # 0.5 - c_ik - t >= 0  => c_ik <= 0.5 - t
        
        # We can add these as a single vectorized constraint function? 
        # SLSQP accepts a dict with 'fun' returning an array.
        
        def boundary_constraints(z):
            if time.time() - start_time > time_limit:
                raise TimeoutError("SLSQP time limit exceeded")
            c = z[:-1].reshape(N, 2)
            t = z[-1]
            # Left/Bottom: c - (t - 0.5) >= 0
            # Right/Top: (0.5 - t) - c >= 0
            # Combine:
            # [c_0x - (t-0.5), c_0y - (t-0.5), ..., (0.5-t)-c_0x, (0.5-t)-c_0y, ...]
            # This is 4N constraints.
            
            lower = c - (t - 0.5)
            upper = (0.5 - t) - c
            return np.concatenate([lower, upper])
        
        cons.append({'type': 'ineq', 'fun': boundary_constraints})
        
        return cons

    # Initial guess
    c_flat = c_init.flatten()
    z0 = np.append(c_flat, t_init)
    
    # Bounds for coordinates
    bounds = [(-0.5, 0.5)] * (2 * N)
    # Bound for t
    bounds.append((0.0, 0.5))
    
    try:
        res = minimize(
            objective, 
            z0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success or res.status == 0:
            c_opt = res.x[:-1].reshape(N, 2)
            t_opt = res.x[-1]
            # Verify validity
            r_val = _compute_radius(c_opt)
            if r_val > 0:
                return c_opt, r_val
        else:
            # Return current best if SLSQP failed
            return c_init, _compute_radius(c_init)
            
    except Exception as e:
        # Return current best if exception
        return c_init, _compute_radius(c_init)

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
    Compute smooth repulsive energy and its gradient.
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
        threshold_d = min_pair_d * 1.1
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
        threshold_w = min_wall_d * 1.1
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
    deadline = start_time + 115

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
    if time.time() < deadline - 70:
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

    # Phase 2: SLSQP NLP with hard constraints
    if time.time() < deadline - 25:
        time_for_nlp = min(50.0, deadline - time.time() - 25)
        t0 = max(best_r * 0.95, 0.03)
        c_nlp, r_nlp = _slsqp_nlp(best_c, t0, time_limit=time_for_nlp)
        if r_nlp > best_r:
            best_r = r_nlp
            best_c = c_nlp.copy()

    # Phase 3: Subgradient polish for fine-tuning
    if time.time() < deadline - 10:
        time_for_polish = min(20.0, deadline - time.time() - 10)
        c_polished, r_polished = _subgradient_polish(best_c, max_iters=3000, time_limit=time_for_polish, base_step=2e-4)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    # Phase 4: Final tiny polish
    if time.time() < deadline - 2:
        time_for_final = min(5.0, deadline - time.time() - 2)
        c_final, r_final = _subgradient_polish(best_c, max_iters=1000, time_limit=time_for_final, base_step=5e-5)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()

    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
