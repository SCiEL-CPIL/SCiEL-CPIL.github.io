"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice (optimized spacing search).
2. Global optimization using Log-Sum-Exp surrogate for max-min distance.
3. High-precision local polish focusing on active constraints.
4. Delaunay-based exact local refinement.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
import time


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_energy(params, beta=200.0):
    """
    Minimize the negative log-sum-exp of distances (not squared).
    This is a smooth approximation of -min(d_i).
    We also add a wall penalty to keep points inside.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise distances (not squared for better approximation of min)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    # Upper triangle indices for pairs
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # Log-Sum-Exp trick for stability
    d_shifted = d_pairs - np.min(d_pairs)
    weights = np.exp(-beta * d_shifted)
    p = weights / np.sum(weights)
    
    # Energy: (1/beta) * log(sum(exp(-beta * d_i)))
    # This is LARGE when distances are small, so minimizing spreads points.
    E_pairs = np.log(np.sum(weights)) / beta
    
    # Gradient computation
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    grad_x[i_idx] -= p * vec_x * inv_d
    grad_x[j_idx] += p * vec_x * inv_d
    
    grad_y[i_idx] -= p * vec_y * inv_d
    grad_y[j_idx] += p * vec_y * inv_d
    
    # Wall Penalty - use soft-min for boundaries too
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
    d_min = np.min(d_all, axis=1, keepdims=True)
    wall_weights = np.exp(-beta * (d_all - d_min))
    wall_weights /= wall_weights.sum(axis=1, keepdims=True)
    
    # Use inverse square penalty for strong repulsion from walls
    eps = 1e-8
    d_left_safe = np.clip(d_left, eps, None)
    d_right_safe = np.clip(d_right, eps, None)
    d_down_safe = np.clip(d_down, eps, None)
    d_up_safe = np.clip(d_up, eps, None)
    
    E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
    
    grad_x += -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
    grad_y += -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
    
    # Combine energies
    E_total = E_wall + E_pairs
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E_total, total_grad


def _initialize_hexagonal():
    """Initialize with hexagonal pattern - optimized search over parameters."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.052, 0.075, 40):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize_energy(c0, beta=100.0):
    """Run L-BFGS-B with Log-Sum-Exp energy."""
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        def fun_and_grad(params):
            return _soft_min_energy(params, beta=beta)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 5000, 'maxls': 100}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0


def _delaunay_refine(c, r_target, max_iters=50):
    """
    Use Delaunay triangulation to find active constraints and refine.
    """
    c = c.copy()
    try:
        tri = Delaunay(c)
    except Exception:
        return c, False
    
    for _ in range(max_iters):
        # Compute circumradii of triangles
        circumradii = []
        for simplex in tri.simplices:
            i, j, k = simplex
            xi, yi = c[i]
            xj, yj = c[j]
            xk, yk = c[k]
            
            # Circumradius formula
            a = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            b = np.sqrt((xi-xk)**2 + (yi-yk)**2)
            d = np.sqrt((xj-xk)**2 + (yj-yk)**2)
            
            # Area using cross product
            area = 0.5 * abs((xj-xi)*(yk-yi) - (yj-yi)*(xk-xi))
            
            if area < 1e-12:
                continue
            
            circumradius = (a * b * d) / (4.0 * area)
            circumradii.append(circumradius)
        
        if not circumradii:
            break
        
        min_circum = min(circumradii)
        
        # Also check boundary distances
        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_bound = np.min(bdists)
        
        # The limiting radius is min(min_circum/2, min_bound)
        # Actually circumradius is the distance from center to vertices,
        # so the min pairwise distance relates to circumradius differently.
        
        # For now, use gradient ascent on the soft objective
        break
    
    return c, True


def _polish_energy(params, beta=300.0):
    """Energy that maximizes the minimum distance using log-sum-exp."""
    c_ = params.reshape(N, 2)
    x = c_[:, 0]
    y = c_[:, 1]
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    bdists = np.minimum(
        np.minimum(0.5 - x, 0.5 + x),
        np.minimum(0.5 - y, 0.5 + y)
    )
    
    d_shifted = d_pairs - np.min(d_pairs)
    w_pairs = np.exp(-beta * d_shifted)
    E_pairs = np.log(np.sum(w_pairs)) / beta
    
    bd_shifted = bdists - np.min(bdists)
    w_bounds = np.exp(-beta * bd_shifted)
    E_bounds = np.log(np.sum(w_bounds)) / beta
    
    E = E_pairs + E_bounds
    
    p_pairs = w_pairs / np.sum(w_pairs)
    p_bounds = w_bounds / np.sum(w_bounds)
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    grad_x[i_idx] -= p_pairs * vec_x * inv_d
    grad_x[j_idx] += p_pairs * vec_x * inv_d
    grad_y[i_idx] -= p_pairs * vec_y * inv_d
    grad_y[j_idx] += p_pairs * vec_y * inv_d
    
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
    d_min = np.min(d_all, axis=1, keepdims=True)
    wall_weights = np.exp(-beta * (d_all - d_min))
    wall_weights /= wall_weights.sum(axis=1, keepdims=True)
    
    grad_x += p_bounds * (-wall_weights[:, 0] + wall_weights[:, 1])
    grad_y += p_bounds * (-wall_weights[:, 2] + wall_weights[:, 3])
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E, total_grad


def _slsqp_active_polish(c, r_target, time_limit=5.0):
    """
    Uses SLSQP to find a configuration where all active constraints are satisfied
    for a target radius r_target.
    """
    start_time = time.time()
    c = c.copy()
    
    tol = 1e-8
    dists = pdist(c, metric='euclidean')
    dist_mat = squareform(dists)
    
    active_mask = dist_mat < (2 * r_target + tol)
    np.fill_diagonal(active_mask, False)
    
    active_pairs = np.argwhere(active_mask)
    active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
    
    bdists = np.minimum(
        np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
        np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
    )
    active_bounds = np.where(bdists < (r_target + tol))[0]
    
    if len(active_pairs) == 0 and len(active_bounds) == 0:
        return c, False
    
    cons = []
    for i, j in active_pairs[:100]:  # Limit to first 100 for speed
        def constraint_pair(x, i=i, j=j):
            xi = x[2*i:2*i+2]
            xj = x[2*j:2*j+2]
            return np.dot(xi-xj, xi-xj) - (2*r_target)**2
        cons.append({'type': 'ineq', 'fun': constraint_pair})
        
    for i in active_bounds[:20]:
        d_left = 0.5 - c[i, 0]
        d_right = c[i, 0] + 0.5
        d_down = 0.5 - c[i, 1]
        d_up = c[i, 1] + 0.5
        min_d = min(d_left, d_right, d_down, d_up)
        
        if d_left == min_d:
            def constraint_wall_left(x, i=i):
                return 0.5 - x[2*i] - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_left})
        elif d_right == min_d:
            def constraint_wall_right(x, i=i):
                return x[2*i] + 0.5 - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_right})
        elif d_down == min_d:
            def constraint_wall_down(x, i=i):
                return 0.5 - x[2*i+1] - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_down})
        else:
            def constraint_wall_up(x, i=i):
                return x[2*i+1] + 0.5 - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_up})

    bounds = [(-0.5 + r_target, 0.5 - r_target)] * (2 * N)
    
    def objective(x):
        return 0.0

    try:
        res = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 50, 'ftol': 1e-12}
        )
        
        if res.success:
            c_new = res.x.reshape(N, 2)
            r_new = _compute_radius(c_new)
            if r_new >= r_target - 1e-10:
                return c_new, True
    except Exception:
        pass
        
    return c, False


def _annealed_polish(c, start_time, time_limit, beta_start=150.0, beta_end=1200.0):
    """
    Multi-temperature annealing polish.
    """
    c = c.copy()
    current_r = _compute_radius(c)
    rng = np.random.RandomState(42)
    
    n_steps = 8
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
        
        try:
            res = minimize(
                lambda p: _polish_energy(p, beta=beta),
                c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.49999, 0.49999)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 600}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
        except Exception:
            pass
        
        if time.time() - start_time > time_limit - 10:
            break
        
        # Small random perturbation
        c_perturbed = c + rng.randn(N, 2) * 0.005
        c_perturbed = np.clip(c_perturbed, -0.49999, 0.49999)
        r_perturbed = _compute_radius(c_perturbed)
        
        if r_perturbed > current_r:
            c = c_perturbed
            current_r = r_perturbed
        else:
            delta_r = current_r - r_perturbed
            T = 0.00008 * (1.0 - (time.time() - start_time) / time_limit)
            if delta_r < T and rng.random() < np.exp(-delta_r / max(T, 1e-10)):
                c = c_perturbed
                current_r = r_perturbed
    
    return c, current_r


def _local_polish(c, time_limit=30.0):
    """
    Fast local polish using active-constraint gradient ascent on radius.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    # Run annealed polish
    if time.time() - start_time < time_limit * 0.75:
        c_annealed, r_annealed = _annealed_polish(c, start_time, time_limit * 0.75, beta_start=150.0, beta_end=1200.0)
        if r_annealed > current_r:
            c = c_annealed
            current_r = r_annealed
    
    # Final high-precision polish
    if time.time() - start_time < time_limit - 5:
        try:
            res = minimize(
                lambda p: _polish_energy(p, beta=2000.0),
                c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.49999, 0.49999)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 5000}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
        except Exception:
            pass
    
    # SLSQP Active-Set Polish
    if time.time() - start_time < time_limit - 1:
        current_r = _compute_radius(c)
        for step in [1e-6, 5e-7, 2e-7, 1e-7]:
            if time.time() - start_time > time_limit - 1:
                break
            target_r = current_r + step
            c_new, success = _slsqp_active_polish(c, target_r, time_limit=5.0)
            if success:
                c = c_new
                current_r = _compute_radius(c)
            else:
                break
    
    return c, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Run global optimization with multiple beta values
    for beta in [40.0, 80.0, 120.0, 180.0]:
        if time.time() > deadline - 90:
            break
        c_opt = _optimize_energy(best_c, beta=beta)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Try multiple perturbed restarts if time allows
    remaining = deadline - time.time() - 70
    if remaining > 25:
        rng = np.random.RandomState(42)
        for attempt in range(3):
            if time.time() > deadline - 70:
                break
            perturbation = rng.randn(N, 2) * 0.008
            c_perturbed = np.clip(best_c + perturbation, -0.4999, 0.4999)
            
            c_opt2 = _optimize_energy(c_perturbed, beta=100.0)
            r_opt2 = _compute_radius(c_opt2)
            
            if r_opt2 > best_r:
                best_r = r_opt2
                best_c = c_opt2.copy()
    
    # Local polish
    remaining = deadline - time.time() - 5
    if remaining > 15:
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 80.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Second polish pass
    remaining = deadline - time.time() - 3
    if remaining > 15:
        c_polished2, r_polished2 = _local_polish(best_c, time_limit=min(remaining, 40.0))
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()
    
    # Final safety check
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
