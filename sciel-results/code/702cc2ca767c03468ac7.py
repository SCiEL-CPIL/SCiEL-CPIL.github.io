"""
Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Strategy:
1. Initialize with multiple diverse patterns (hexagonal variants, grid).
2. Minimize a soft-min surrogate energy via L-BFGS-B with annealed temperature.
3. Refine via improved local search with active-set approach.
4. Final ultra-precise polish with tight tolerances.
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

def _compute_radius_fast(c):
    """Fast radius computation using only nearest-neighbor info when available."""
    n = c.shape[0]
    if n == 0:
        return 0.0

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if n < 2:
        return min_boundary

    # Use cdist for speed
    from scipy.spatial.distance import cdist
    dm = cdist(c, c, metric='euclidean')
    np.fill_diagonal(dm, np.inf)
    min_pair_dist = np.min(dm)

    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_energy(params, temperature=1.0):
    """
    Compute soft-min surrogate energy and its gradient.
    Uses log-sum-exp to approximate min distance.
    Energy = -log(sum_{i<j} exp(-d_ij/(2*temp))) - log(sum_i exp(-d_iw/temp))
    This is a smooth approximation of -min(min_pair_dist/2, min_boundary)
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    # Pair distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1e-10)
    d = np.sqrt(d2)
    
    # Wall distances
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = y + 0.5
    
    # Clip to avoid numerical issues
    d = np.maximum(d, 1e-10)
    d_left = np.maximum(d_left, 1e-10)
    d_right = np.maximum(d_right, 1e-10)
    d_down = np.maximum(d_down, 1e-10)
    d_up = np.maximum(d_up, 1e-10)
    
    # Soft-min of pair distances (we want to maximize min, so minimize negative soft-min)
    # For pairs: only upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = d[mask]
    
    # Numerically stable log-sum-exp
    max_d_pair = np.max(d_pairs)
    log_sum_exp_pairs = max_d_pair / (2*temperature) + np.log(np.sum(np.exp(-(d_pairs - max_d_pair)/(2*temperature))))
    E_pairs = temperature * log_sum_exp_pairs
    
    # For walls
    d_walls = np.concatenate([d_left, d_right, d_down, d_up])
    max_d_wall = np.max(d_walls)
    log_sum_exp_walls = max_d_wall / temperature + np.log(np.sum(np.exp(-(d_walls - max_d_wall)/temperature)))
    E_walls = temperature * log_sum_exp_walls
    
    # Total energy (we minimize this, which corresponds to maximizing the soft-min radius)
    total_E = -(E_pairs + E_walls)
    
    # Gradient
    # Pair gradient
    exp_pairs = np.exp(-(d_pairs - max_d_pair)/(2*temperature))
    
    # Reconstruct full distance matrix for gradient
    d_full = d.copy()
    np.fill_diagonal(d_full, 1.0)
    exp_full = np.exp(-(d_full - max_d_pair)/(2*temperature))
    np.fill_diagonal(exp_full, 0.0)
    sum_exp_full = np.sum(exp_full, axis=1)
    
    # grad_x_i = -0.5 * sum_j [exp(-d_ij/(2*temp)) * (x_i - x_j)/d_ij] / sum_j exp(-d_ij/(2*temp))
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    grad_x -= 0.5 * (dx * exp_full / d_full).sum(axis=1) / sum_exp_full
    grad_y -= 0.5 * (dy * exp_full / d_full).sum(axis=1) / sum_exp_full
    
    # Wall gradient
    exp_walls = np.exp(-(d_walls - max_d_wall)/temperature)
    sum_exp_walls = np.sum(exp_walls)
    
    exp_left = exp_walls[:N]
    exp_right = exp_walls[N:2*N]
    exp_down = exp_walls[2*N:3*N]
    exp_up = exp_walls[3*N:]
    
    grad_x += (exp_left - exp_right) / sum_exp_walls
    grad_y += (exp_down - exp_up) / sum_exp_walls
    
    total_grad = np.concatenate([grad_x, grad_y])
    
    return total_E, total_grad

def _slsqp_polish(c, time_limit=20.0):
    """
    SLSQP-based active-set refinement.
    Maximizes t subject to:
      |c_i - c_j|^2 >= 4*t^2 for all i<j
      0.5 - |c_ik| >= t for all i,k
    Using a subset of active constraints identified from current solution.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    n_vars = 2 * N
    
    for round_num in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Identify active constraints
        dist_matrix = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        
        # Active pairs: those within 2% of min
        min_pair = np.min(dist_matrix)
        active_pairs = np.argwhere(dist_matrix < min_pair * 1.02)
        # Remove duplicates
        unique_pairs = set()
        for i, j in active_pairs:
            if i < j:
                unique_pairs.add((i, j))
            elif j < i:
                unique_pairs.add((j, i))
        
        # Active boundaries
        min_bd = np.min(bdists)
        active_bounds = np.where(bdists < min_bd * 1.02)[0]
        
        if len(unique_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Build constraint functions for SLSQP
        pair_list = list(unique_pairs)
        
        def objective(x):
            # Minimize -t (maximize t)
            return -x[-1]
        
        def objective_grad(x):
            g = np.zeros(n_vars + 1)
            g[-1] = -1.0
            return g
        
        def constraint_func(x, ci, cj):
            # |c_i - c_j|^2 - 4*t^2 >= 0
            xi = x[2*ci:2*ci+2]
            xj = x[2*cj:2*cj+2]
            t = x[-1]
            return np.sum((xi - xj)**2) - 4*t*t
        
        def constraint_jac(x, ci, cj):
            # Jacobian row for constraint on pair (ci, cj)
            jrow = np.zeros(n_vars + 1)
            xi = x[2*ci:2*ci+2]
            xj = x[2*cj:2*cj+2]
            t = x[-1]
            diff = xi - xj
            jrow[2*ci:2*ci+2] = 2 * diff
            jrow[2*cj:2*cj+2] = -2 * diff
            jrow[-1] = -8 * t
            return jrow
        
        def boundary_constraint(x, idx, side):
            # 0.5 - |c_ik| - t >= 0
            # side: 0=left, 1=right, 2=down, 3=up
            coord = x[2*idx + (0 if side < 2 else 1)]
            if side == 0:  # left: 0.5 - x >= t
                return 0.5 - coord - x[-1]
            elif side == 1:  # right: 0.5 + x >= t
                return 0.5 + coord - x[-1]
            elif side == 2:  # down: 0.5 - y >= t
                return 0.5 - coord - x[-1]
            else:  # up: 0.5 + y >= t
                return 0.5 + coord - x[-1]
        
        def boundary_jac(x, idx, side):
            jrow = np.zeros(n_vars + 1)
            coord_idx = 2*idx + (0 if side < 2 else 1)
            if side in [0, 2]:  # left or down
                jrow[coord_idx] = -1.0
            else:  # right or up
                jrow[coord_idx] = 1.0
            jrow[-1] = -1.0
            return jrow
        
        # Build constraints list
        constraints = []
        for ci, cj in pair_list:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, ci=ci, cj=cj: constraint_func(x, ci, cj),
                'jac': lambda x, ci=ci, cj=cj: constraint_jac(x, ci, cj)
            })
        
        for idx in active_bounds:
            x, y = c[idx]
            d_left = 0.5 - x
            d_right = x + 0.5
            d_down = 0.5 - y
            d_up = y + 0.5
            min_d = min(d_left, d_right, d_down, d_up)
            if min_d == d_left:
                side = 0
            elif min_d == d_right:
                side = 1
            elif min_d == d_down:
                side = 2
            else:
                side = 3
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=idx, side=side: boundary_constraint(x, idx, side),
                'jac': lambda x, idx=idx, side=side: boundary_jac(x, idx, side)
            })
        
        # Initial guess
        x0 = np.concatenate([c.flatten(), [current_r]])
        
        bounds = [(-0.499, 0.499)] * (2 * N) + [(0.0, 0.5)]
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success and res.fun < -best_r + 1e-15:
                new_c = res.x[:2*N].reshape(N, 2)
                new_r = _compute_radius(new_c)
                if new_r > best_r:
                    best_r = new_r
                    best_c = new_c.copy()
                    c = new_c.copy()
                    current_r = new_r
        
        except Exception:
            pass
    
    return best_c, best_r

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

def _initialize_hexagonal_optimized():
    """Initialize with optimized hexagonal pattern for 200 circles."""
    # Try different aspect ratios and spacings
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.068, 0.072, 5):
        c = _initialize_hexagonal(d)
        if c is not None:
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    return best_c

def _initialize_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _initialize_random(seed=42):
    """Initialize with random points."""
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.45, 0.45, size=(N, 2))
    return c

def _local_polish(c, max_iters=5000, time_limit=25.0, base_step=1e-4):
    """
    Improved local search to maximize minimum distance.
    Uses active-set approach with incremental distance updates for speed.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    step = base_step
    stall_count = 0
    max_stall = 80

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        # Compute distance matrix efficiently
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(c, c, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)

        min_val = np.min(dist_matrix)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        i, j = min_idx

        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_boundary_val = np.min(bdists)
        b_idx = np.argmin(bdists)

        improved = False

        if min_val / 2.0 < min_boundary_val:
            # Move i and j apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                unit_vec = vec / norm

                # Try multiple step sizes
                for s in [step, step * 0.5, step * 1.5]:
                    new_ci = c[i] + unit_vec * s
                    new_cj = c[j] - unit_vec * s

                    if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                        old_ci, old_cj = c[i].copy(), c[j].copy()
                        c[i] = new_ci
                        c[j] = new_cj
                        new_r = _compute_radius_fast(c)
                        if new_r > current_r + 1e-15:
                            current_r = new_r
                            improved = True
                            break
                        else:
                            c[i] = old_ci
                            c[j] = old_cj

                if not improved:
                    # Try random perturbation of the pair
                    for _ in range(3):
                        rand_dir = np.random.randn(2)
                        rand_dir /= np.linalg.norm(rand_dir)
                        new_ci = c[i] + rand_dir * step * 0.5
                        new_cj = c[j] - rand_dir * step * 0.5
                        if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                            old_ci, old_cj = c[i].copy(), c[j].copy()
                            c[i] = new_ci
                            c[j] = new_cj
                            new_r = _compute_radius_fast(c)
                            if new_r > current_r + 1e-15:
                                current_r = new_r
                                improved = True
                                break
                            else:
                                c[i] = old_ci
                                c[j] = old_cj

        else:
            # Move boundary circle inwards
            x, y = c[b_idx]
            d_left = 0.5 - x
            d_right = x + 0.5
            d_down = 0.5 - y
            d_up = y + 0.5

            min_d = min(d_left, d_right, d_down, d_up)

            for s in [step, step * 0.5, step * 1.5]:
                if min_d == d_left:
                    new_pt = c[b_idx] + np.array([s, 0])
                elif min_d == d_right:
                    new_pt = c[b_idx] + np.array([-s, 0])
                elif min_d == d_down:
                    new_pt = c[b_idx] + np.array([0, s])
                else:
                    new_pt = c[b_idx] + np.array([0, -s])

                if np.all(np.abs(new_pt) < 0.5):
                    old_pt = c[b_idx].copy()
                    c[b_idx] = new_pt
                    new_r = _compute_radius_fast(c)
                    if new_r > current_r + 1e-15:
                        current_r = new_r
                        improved = True
                        break
                    else:
                        c[b_idx] = old_pt

        if improved:
            stall_count = 0
            step = min(step * 1.1, 1e-3)
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        else:
            stall_count += 1
            step = max(step * 0.7, 1e-7)
            if stall_count > max_stall:
                break

    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 142  # Safety margin before 150s kill

    # Generate multiple initializations
    inits = []

    # Hexagonal with different spacings
    for d in np.linspace(0.065, 0.075, 5):
        c_hex = _initialize_hexagonal(d)
        if c_hex is not None:
            inits.append(c_hex)

    # Optimized hexagonal
    c_hex_opt = _initialize_hexagonal_optimized()
    if c_hex_opt is not None:
        inits.append(c_hex_opt)

    # Grid
    inits.append(_initialize_grid())

    # Random with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        inits.append(_initialize_random(seed))

    # Evaluate each initialization and pick the best few
    init_results = []
    for c_init in inits:
        c_init = np.clip(c_init, -0.49, 0.49)
        r_init = _compute_radius(c_init)
        init_results.append((r_init, c_init))

    # Sort by radius and take top 4
    init_results.sort(key=lambda x: x[0], reverse=True)
    top_inits = [c for r, c in init_results[:4]]

    best_c = top_inits[0].copy()
    best_r = _compute_radius(best_c)

    # Run L-BFGS-B with annealed temperature on the best initialization
    if time.time() < deadline - 55:
        try:
            bounds = [(-0.499, 0.499)] * (2 * N)
            
            # Multi-temperature annealing schedule - tighter for better convergence
            temperatures = [0.04, 0.02, 0.01, 0.005, 0.003]
            current_params = best_c.flatten()
            
            for temp in temperatures:
                if time.time() > deadline - 55:
                    break
                    
                max_iter = 300 if temp >= 0.01 else 500
                res = minimize(
                    lambda x: _soft_min_energy(x, temperature=temp),
                    current_params,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
                )
                
                c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
                r_opt = _compute_radius(c_opt)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt.copy()
                    current_params = c_opt.flatten()

        except Exception:
            pass

    # If time permits, try L-BFGS-B on the second-best init too
    if time.time() < deadline - 40 and len(top_inits) > 1:
        try:
            c2 = top_inits[1]
            bounds = [(-0.499, 0.499)] * (2 * N)
            
            temperatures = [0.02, 0.01, 0.005]
            current_params = c2.flatten()
            
            for temp in temperatures:
                if time.time() > deadline - 40:
                    break
                    
                res2 = minimize(
                    lambda x: _soft_min_energy(x, temperature=temp),
                    current_params,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1500}
                )
                
                c_opt2 = np.clip(res2.x.reshape(N, 2), -0.499, 0.499)
                r_opt2 = _compute_radius(c_opt2)
                
                if r_opt2 > best_r:
                    best_r = r_opt2
                    best_c = c_opt2.copy()
                    current_params = c_opt2.flatten()

        except Exception:
            pass

    # SLSQP active-set refinement
    if time.time() < deadline - 20:
        try:
            time_slsqp = min(18.0, deadline - time.time() - 20)
            c_slsqp, r_slsqp = _slsqp_polish(best_c, time_limit=time_slsqp)
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_c = c_slsqp.copy()
        except Exception:
            pass

    # Local polish
    if time.time() < deadline - 10:
        time_for_polish = min(15.0, deadline - time.time() - 10)
        c_polished, r_polished = _local_polish(best_c, max_iters=5000, time_limit=time_for_polish)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    # Final ultra-precise polish with smaller steps
    if time.time() < deadline - 5:
        time_for_final = min(8.0, deadline - time.time() - 5)
        c_final, r_final = _local_polish(best_c, max_iters=3000, time_limit=time_for_final, base_step=1e-5)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
