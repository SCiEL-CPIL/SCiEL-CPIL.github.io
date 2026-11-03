"""
Pack 200 equal circles in a unit square.

Strategy:
1. Multi-start: generate several initial configurations (hex lattices with
   slightly different spacing/phase, plus a jittered variant).
2. For each start, run a force-directed relaxation to remove overlaps.
3. Then run a smooth soft-min objective with L-BFGS-B, increasing the
   sharpness parameter beta to sharpen the min.
4. Use a projected-gradient-descent / local-search phase to escape local
   minima, with adaptive step size and random perturbations.
5. Final high-precision polish with very high beta.
6. Track the best radius found so far and return the best configuration.

We keep the public function `construct()` returning an (N, 2) array.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 149  # safety margin before the 150s kill

def _get_radius(pts):
    """
    Compute the maximum common radius for a set of centers:
    min of (distance to walls) and (half of minimum pairwise distance).
    """
    if len(pts) == 0:
        return 0.0

    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)

    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0

    return min(min_wall, min_pair)

def _soft_min(vals, beta):
    """
    Soft minimum: -1/beta * log(sum(exp(-beta * x_i))).
    Numerically stable via max shift.
    """
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    """
    Objective for minimization: -soft_min(constraints).
    Constraints:
      - 4 wall distances per point
      - pairwise distances / 2
    Returns (value, gradient).
    """
    pts = x.reshape((N, 2))

    # Wall values
    C_wall = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])

    # Pair values
    i_idx, j_idx = np.triu_indices(N, k=1)
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    C_pair = pair_dists / 2.0

    # Combine all constraint values
    C = np.concatenate([C_wall, C_pair])

    # Soft min
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val

    # Weights
    weights = exp_shifted / sum_exp

    # Gradient
    grad_pts = np.zeros((N, 2))

    # Wall gradients
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)

    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top

    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)

    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)

    grad_pair_x_i = dx_safe / (2.0 * dists_safe)
    grad_pair_x_j = -dx_safe / (2.0 * dists_safe)
    grad_pair_y_i = dy_safe / (2.0 * dists_safe)
    grad_pair_y_j = -dy_safe / (2.0 * dists_safe)

    grad_pair_x_i = np.where(valid, grad_pair_x_i, 0.0)
    grad_pair_x_j = np.where(valid, grad_pair_x_j, 0.0)
    grad_pair_y_i = np.where(valid, grad_pair_y_i, 0.0)
    grad_pair_y_j = np.where(valid, grad_pair_y_j, 0.0)

    pair_weights = weights[4*N:]

    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)

    grad_pts = -grad_pts / beta

    return obj_val, grad_pts.flatten()

def _generate_hex_initial(spacing, phase_shift=0.0, margin=0.001):
    """
    Generate a hexagonal lattice packing with given spacing and phase shift.
    Returns N points inside [-0.5, 0.5]^2.
    """
    points = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5 + row_h / 2 + phase_shift
    row_idx = 0
    while y < 0.5:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + spacing / 2 + shift + (phase_shift if row_idx % 2 == 0 else 0.0)
        while x < 0.5:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]

    if len(pts) < N:
        # Fallback: random
        return np.random.uniform(-0.4, 0.4, (N, 2))

    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]

    return pts

def _generate_jittered_hex(spacing, jitter=0.002, seed=0):
    """
    Generate hex lattice with small random jitter to break symmetry.
    """
    rng = np.random.RandomState(seed)
    pts = _generate_hex_initial(spacing, phase_shift=0.0)
    pts = pts + rng.uniform(-jitter, jitter, size=pts.shape)
    pts = np.clip(pts, -0.499, 0.499)
    return pts

def _force_directed_phase(pts, time_limit, iterations=None):
    """
    Force-directed repulsion to spread out overlapping circles.
    Uses vectorized inverse-square repulsion between close pairs and boundary repulsion.
    """
    if iterations is None:
        iterations = max(500, int(time_limit * 150))

    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    step = 0.004
    lr_decay = 0.998

    for it in range(iterations):
        if time.time() - t_start > time_limit:
            break

        forces = np.zeros_like(pts)

        # Vectorized pairwise force computation
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)

        threshold = 0.12
        close_mask = (D < threshold) & (D > 1e-10)

        # Compute forces: for each pair (i,j) with D[i,j] < threshold,
        # force on i is proportional to (pts[i] - pts[j]) / D[i,j]^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        inv_d3 = np.where(close_mask, 1.0 / (D**3 + 1e-9), 0.0)
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)

        # Boundary repulsion
        margin = 0.05
        forces[:, 0] += np.where(pts[:, 0] < -0.5 + margin, (margin - (pts[:, 0] + 0.5)) * 3.0, 0.0)
        forces[:, 0] -= np.where(pts[:, 0] > 0.5 - margin, (pts[:, 0] - (0.5 - margin)) * 3.0, 0.0)
        forces[:, 1] += np.where(pts[:, 1] < -0.5 + margin, (margin - (pts[:, 1] + 0.5)) * 3.0, 0.0)
        forces[:, 1] -= np.where(pts[:, 1] > 0.5 - margin, (pts[:, 1] - (0.5 - margin)) * 3.0, 0.0)

        pts += step * forces
        pts = np.clip(pts, -0.5, 0.5)

        step *= lr_decay

        r = _get_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_r, best_pts

def _projected_gradient_descent(pts, time_limit, beta=1000, max_iter=2000):
    """
    Projected gradient descent to maximize the minimum radius.
    Uses the soft-min objective gradient.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    
    step = 0.001
    decay = 0.999
    min_step = 1e-6
    
    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break
        
        # Compute gradient of soft-min objective
        x = pts.flatten()
        _, grad = _objective_and_grad(x, beta)
        grad_pts = grad.reshape((N, 2))
        
        # Projected gradient ascent (we want to maximize, so go against gradient)
        pts_new = pts + step * grad_pts
        pts_new = np.clip(pts_new, -0.5, 0.5)
        
        r = _get_radius(pts_new)
        if r > best_r:
            best_r = r
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Reduce step size if no improvement
            step *= 0.9
            if step < min_step:
                break
        
        step = max(step * decay, min_step)
    
    return best_r, best_pts

def _perturb_and_optimize(pts, t_start, time_budget, rng):
    """
    Perturb the configuration by moving the most constrained points,
    then run gradient optimization.
    """
    r_curr = _get_radius(pts)
    
    # Find the most constrained pairs
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    
    # Find pairs that are limiting the radius
    limiting_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if D[i, j] / 2.0 <= r_curr + 1e-9:
                limiting_pairs.append((i, j, D[i, j]))
    
    # Also check walls
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if len(limiting_pairs) == 0:
        # No limiting pairs, try random perturbation
        pts_new = pts.copy()
        idx = rng.integers(0, N, size=5)
        pts_new[idx] += rng.normal(0, 0.005, (5, 2))
        pts_new = np.clip(pts_new, -0.5, 0.5)
    else:
        # Perturb points involved in limiting pairs
        pts_new = pts.copy()
        involved = set()
        for i, j, d in limiting_pairs[:10]:  # Top 10 most limiting
            involved.add(i)
            involved.add(j)
        
        if len(involved) == 0:
            involved = {0}
        
        for idx in involved:
            pts_new[idx] += rng.normal(0, 0.003, 2)
        
        pts_new = np.clip(pts_new, -0.5, 0.5)
    
    # Run gradient optimization on the perturbed configuration
    r_opt, pts_opt = _run_gradient_opt(pts_new, t_start, time_budget)
    
    if r_opt > r_curr:
        return r_opt, pts_opt
    else:
        return r_curr, pts

def _run_gradient_opt(pts, t_start, time_budget):
    """
    Run gradient-based optimization with increasing beta.
    """
    if time.time() - t_start > time_budget:
        return _get_radius(pts), pts
    
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    # Force-directed phase to fix initial overlaps
    fd_time = min(3.0, time_budget * 0.1)
    if fd_time > 0.5:
        r_fd, pts_fd = _force_directed_phase(pts, time_limit=fd_time)
        if r_fd > best_r:
            best_r = r_fd
            best_x = pts_fd.flatten()
            x0 = pts_fd.flatten()
    
    # Gradient optimization with increasing beta
    betas = [20, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        elapsed = time.time() - t_start
        if elapsed > time_budget - 2.0:
            break
        
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
        
        try:
            maxiter = 100 if beta < 100 else 200
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - t_start
    remaining = time_budget - (time.time() - t_start)
    if remaining > 2:
        for beta in [1000, 2000, 5000]:
            if time.time() - t_start > time_budget - 1:
                break
            try:
                def fun_and_grad(x):
                    val, g = _objective_and_grad(x, beta)
                    return val, g
                
                res = minimize(
                    fun_and_grad, 
                    x0, 
                    jac=True, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 100}
                )
                x0 = res.x
                pts_curr = x0.reshape((N, 2))
                pts_curr = np.clip(pts_curr, -0.5, 0.5)
                
                r = _get_radius(pts_curr)
                if r > best_r:
                    best_r = r
                    best_x = x0.copy()
            except Exception:
                pass
    
    return best_r, best_x.reshape((N, 2))

def construct():
    t0 = time.time()
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_r = 0.0
    best_pts = None
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Hex lattice with different spacings
    for spacing in [0.0732, 0.0720, 0.0710]:
        pts_hex = _generate_hex_initial(spacing)
        initial_configs.append(pts_hex)
    
    # Jittered hex lattices
    for seed in [0, 1, 2]:
        pts_jitter = _generate_jittered_hex(0.0732, jitter=0.002, seed=seed)
        initial_configs.append(pts_jitter)
    
    # Random initial configuration
    pts_random = np.random.uniform(-0.4, 0.4, (N, 2))
    initial_configs.append(pts_random)
    
    # Try each initial configuration
    time_per_config = (_TIME_LIMIT - 10) / len(initial_configs)
    
    for idx, pts_init in enumerate(initial_configs):
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining < time_per_config * 0.5:
            break
        
        # Force-directed phase
        fd_time = min(3.0, remaining * 0.3)
        r_fd, pts_fd = _force_directed_phase(pts_init, time_limit=fd_time)
        
        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd
        
        # Gradient optimization
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        opt_time = min(remaining * 0.6, time_per_config * 0.6)
        
        if opt_time > 2:
            r_opt, pts_opt = _run_gradient_opt(pts_fd, t0, opt_time)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt
        
        # Perturbation and re-optimization
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining > time_per_config * 0.3:
            r_pert, pts_pert = _perturb_and_optimize(
                best_pts, t0, remaining * 0.7, rng
            )
            if r_pert > best_r:
                best_r = r_pert
                best_pts = pts_pert
    
    # Final high-precision polish
    elapsed = time.time() - t0
    remaining = _TIME_LIMIT - 3 - elapsed
    if remaining > 5 and best_pts is not None:
        x0 = best_pts.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        for beta in [1000, 2000, 5000, 10000]:
            if time.time() - t0 > _TIME_LIMIT - 3:
                break
            
            def fun_and_grad(x):
                val, g = _objective_and_grad(x, beta)
                return val, g
            
            try:
                res = minimize(
                    fun_and_grad, 
                    x0, 
                    jac=True, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 200}
                )
                x0 = res.x
                pts_curr = x0.reshape((N, 2))
                pts_curr = np.clip(pts_curr, -0.5, 0.5)
                
                r = _get_radius(pts_curr)
                if r > best_r:
                    best_r = r
                    best_pts = pts_curr
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
