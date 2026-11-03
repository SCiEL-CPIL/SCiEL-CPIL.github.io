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
5. Track the best radius found so far and return the best configuration.

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
        iterations = max(300, int(time_limit * 100))

    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    step = 0.003
    lr_decay = 0.999

    for it in range(iterations):
        if time.time() - t_start > time_limit:
            break

        forces = np.zeros_like(pts)

        # Vectorized pairwise force computation
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)

        threshold = 0.10
        close_mask = (D < threshold) & (D > 1e-10)

        # Compute forces: for each pair (i,j) with D[i,j] < threshold,
        # force on i is proportional to (pts[i] - pts[j]) / D[i,j]^2
        # We can do this with matrix operations
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        # Only apply where close_mask is True
        force_mag = np.where(close_mask, 1.0 / (D + 1e-6), 0.0)
        # force on i from j: (pts[i] - pts[j]) * force_mag[i,j] / D[i,j]
        # Actually: direction is (pts[i]-pts[j])/D, magnitude is 1/D^2, so total = (pts[i]-pts[j])/D^3
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
    Projected gradient descent on the soft-min objective.
    Uses adaptive step size and random restarts to escape local minima.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    step_size = 0.001
    patience = 50
    no_improve_count = 0

    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break

        # Compute gradient of -soft_min
        x = pts.flatten()
        _, grad = _objective_and_grad(x, beta)
        grad_pts = grad.reshape((N, 2))

        # Projected gradient step
        new_pts = pts - step_size * grad_pts
        new_pts = np.clip(new_pts, -0.5, 0.5)

        # Check improvement
        new_r = _get_radius(new_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            step_size = min(step_size * 1.05, 0.005)
        else:
            no_improve_count += 1
            step_size *= 0.95

            # Random perturbation to escape local minima
            if no_improve_count >= patience:
                pts = pts + np.random.randn(N, 2) * 0.005
                pts = np.clip(pts, -0.5, 0.5)
                no_improve_count = 0
                step_size = 0.001

    return best_r, best_pts

def _epigraph_objective_and_grad(x, beta=5000):
    """
    Epigraph formulation: minimize -t subject to t <= c_i for all constraints.
    We use a smooth approximation: minimize -soft_min(c_i) + penalty for t > soft_min.
    Actually, we'll directly maximize the minimum constraint value using a smooth approach.
    
    Alternative: Use the fact that at optimum, t = min(c_i). We can use L-BFGS-B
    with a very high beta to get close to the true min, then do a final local search
    on the active constraints.
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
    
    # Very high beta soft min
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

def _final_polish(pts, time_limit):
    """
    Final high-precision polish using multiple stages:
    1. L-BFGS-B with very high beta (10000+)
    2. Active set identification and local refinement
    3. Tiny random perturbations with re-optimization
    """
    t_start = time.time()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Stage 1: L-BFGS-B with very high beta
    for beta in [5000, 10000, 20000]:
        elapsed = time.time() - t_start
        if elapsed > time_limit * 0.6:
            break
        
        x0 = best_pts.flatten()
        
        def fun_and_grad(x):
            val, g = _epigraph_objective_and_grad(x, beta)
            return val, g
        
        try:
            res = minimize(
                fun_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 200}
            )
            pts_curr = res.x.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_pts = pts_curr.copy()
        except Exception:
            pass
    
    # Stage 2: Identify active constraints and do local refinement
    elapsed = time.time() - t_start
    if elapsed < time_limit * 0.8:
        pts_curr = best_pts.copy()
        r_curr = best_r
        
        # Find the minimum constraint value
        C_wall = np.concatenate([
            pts_curr[:, 0] + 0.5,
            0.5 - pts_curr[:, 0],
            pts_curr[:, 1] + 0.5,
            0.5 - pts_curr[:, 1]
        ])
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        D = cdist(pts_curr, pts_curr)
        pair_dists = D[i_idx, j_idx]
        C_pair = pair_dists / 2.0
        
        C_all = np.concatenate([C_wall, C_pair])
        min_val = np.min(C_all)
        
        # Find all constraints within 1e-6 of the minimum (active set)
        active_tol = max(1e-6, min_val * 1e-4)
        active_mask = C_all < (min_val + active_tol)
        
        # For each active constraint, try small perturbations
        n_active = np.sum(active_mask)
        if n_active > 0 and n_active < 50:
            # Try moving points involved in active constraints
            rng = np.random.RandomState(42)
            for _ in range(20):
                if time.time() - t_start > time_limit:
                    break
                
                # Pick a random active constraint and perturb the involved points
                active_indices = np.where(active_mask)[0]
                chosen = rng.choice(active_indices)
                
                # Determine which points are involved
                if chosen < 4 * N:
                    # Wall constraint
                    wall_idx = chosen // N
                    point_idx = chosen % N
                    # Perturb this point
                    delta = rng.randn(2) * 0.0001
                    new_pts = pts_curr.copy()
                    new_pts[point_idx] += delta
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                else:
                    # Pair constraint
                    pair_idx = chosen - 4 * N
                    i = i_idx[pair_idx]
                    j = j_idx[pair_idx]
                    # Perturb both points
                    delta_i = rng.randn(2) * 0.0001
                    delta_j = rng.randn(2) * 0.0001
                    new_pts = pts_curr.copy()
                    new_pts[i] += delta_i
                    new_pts[j] += delta_j
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                
                new_r = _get_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
    
    return best_r, best_pts

def _run_optimization(pts, time_limit, betas=None):
    """
    Full optimization pipeline for a single initial configuration:
    1. Force-directed relaxation
    2. L-BFGS-B with increasing beta
    3. Projected gradient descent polish
    """
    if betas is None:
        betas = [10, 20, 50, 100, 200, 500, 1000, 2000]

    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)

    best_r = _get_radius(pts)
    best_x = x0.copy()

    # Phase 1: Force-directed
    fd_time = min(5.0, time_limit * 0.15)
    if fd_time > 1.0:
        r_fd, pts_fd = _force_directed_phase(pts, time_limit=fd_time)
        if r_fd > best_r:
            best_r = r_fd
            best_x = pts_fd.flatten()
            x0 = pts_fd.flatten()

    # Phase 2: L-BFGS-B with increasing beta
    for beta in betas:
        elapsed = time.time() - t_start
        if elapsed > time_limit:
            break
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break

        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g

        try:
            # Scale maxiter based on remaining time and beta
            if beta >= 500:
                maxiter = 200
            elif beta >= 100:
                maxiter = 150
            else:
                maxiter = 100
            res = minimize(
                fun_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100}
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

    # Phase 3: Projected gradient descent
    elapsed = time.time() - t_start
    remaining = time_limit - elapsed
    if remaining > 3.0:
        r_pgd, pts_pgd = _projected_gradient_descent(
            best_x.reshape((N, 2)),
            time_limit=remaining * 0.7,
            beta=2000,
            max_iter=2000
        )
        if r_pgd > best_r:
            best_r = r_pgd
            best_x = pts_pgd.flatten()

    return best_r, best_x.reshape((N, 2))

def _active_set_local_search(pts, time_limit, max_iter=500):
    """
    Local search that focuses on the active constraints (those with the minimum value).
    Uses a more aggressive strategy: identify the bottleneck constraints and try to
    move the involved points in directions that increase the minimum.
    """
    t_start = time.time()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(123)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break
        
        pts_curr = best_pts.copy()
        r_curr = best_r
        
        # Compute all constraint values
        C_wall = np.concatenate([
            pts_curr[:, 0] + 0.5,
            0.5 - pts_curr[:, 0],
            pts_curr[:, 1] + 0.5,
            0.5 - pts_curr[:, 1]
        ])
        
        D = cdist(pts_curr, pts_curr)
        pair_dists = D[i_idx, j_idx]
        C_pair = pair_dists / 2.0
        
        C_all = np.concatenate([C_wall, C_pair])
        min_val = np.min(C_all)
        
        # Find active constraints (within tolerance of min)
        active_tol = max(1e-7, min_val * 1e-5)
        active_mask = C_all < (min_val + active_tol)
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            break
        
        # For each active constraint, try to improve it
        improved = False
        for _ in range(10):  # Try a few random active constraints per iteration
            if time.time() - t_start > time_limit:
                break
            
            chosen = rng.choice(active_indices)
            
            if chosen < 4 * N:
                # Wall constraint
                wall_type = chosen // N  # 0: left, 1: right, 2: bottom, 3: top
                point_idx = chosen % N
                
                # Try to move the point away from the wall
                direction = np.zeros(2)
                if wall_type == 0:  # left wall: x is too small
                    direction[0] = 1.0
                elif wall_type == 1:  # right wall: x is too large
                    direction[0] = -1.0
                elif wall_type == 2:  # bottom wall: y is too small
                    direction[1] = 1.0
                else:  # top wall: y is too large
                    direction[1] = -1.0
                
                # Try multiple step sizes
                for step in [0.0001, 0.0005, 0.001, 0.002]:
                    new_pts = pts_curr.copy()
                    new_pts[point_idx] += step * direction
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    
                    new_r = _get_radius(new_pts)
                    if new_r > best_r:
                        best_r = new_r
                        best_pts = new_pts.copy()
                        improved = True
                        break
            else:
                # Pair constraint
                pair_idx = chosen - 4 * N
                i = i_idx[pair_idx]
                j = j_idx[pair_idx]
                
                # Direction to separate the two points
                diff = pts_curr[i] - pts_curr[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                direction_i = diff / dist
                direction_j = -diff / dist
                
                # Try multiple step sizes
                for step in [0.0001, 0.0005, 0.001, 0.002]:
                    new_pts = pts_curr.copy()
                    new_pts[i] += step * direction_i
                    new_pts[j] += step * direction_j
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    
                    new_r = _get_radius(new_pts)
                    if new_r > best_r:
                        best_r = new_r
                        best_pts = new_pts.copy()
                        improved = True
                        break
        
        if not improved and it > 100:
            # Add a small random perturbation to escape
            pts_curr = best_pts + rng.randn(N, 2) * 0.001
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            r_curr = _get_radius(pts_curr)
            if r_curr > best_r:
                best_r = r_curr
                best_pts = pts_curr.copy()
    
    return best_r, best_pts

def construct():
    t0 = time.time()
    np.random.seed(42)

    best_r = 0.0
    best_pts = None

    # Generate multiple initial configurations
    inits = []

    # Init 1: Standard hex with spacing ~0.0732
    pts1 = _generate_hex_initial(0.0732, phase_shift=0.0)
    inits.append(pts1)

    # Init 2: Shifted hex
    pts2 = _generate_hex_initial(0.0732, phase_shift=0.01)
    inits.append(pts2)

    # Init 3: Slightly denser hex (will have overlaps, optimizer fixes)
    pts3 = _generate_hex_initial(0.070, phase_shift=0.0)
    inits.append(pts3)

    # Init 4: Jittered hex
    pts4 = _generate_jittered_hex(0.0732, jitter=0.002, seed=0)
    inits.append(pts4)

    # Init 5: Jittered hex with different seed
    pts5 = _generate_jittered_hex(0.0732, jitter=0.003, seed=1)
    inits.append(pts5)

    num_inits = len(inits)
    time_per_init = (_TIME_LIMIT - 25) / num_inits  # reserve 25s for final polish

    for i, init_pts in enumerate(inits):
        elapsed = time.time() - t0
        remaining_for_phase = _TIME_LIMIT - 25 - elapsed
        if remaining_for_phase <= 5:
            break

        t_limit = min(time_per_init, remaining_for_phase)
        r_opt, pts_opt = _run_optimization(init_pts, time_limit=t_limit)

        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt

    # Final polish with high-beta optimization
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 3 - elapsed
        if remaining_time > 5:
            # First do a standard optimization pass
            betas_polish = [1000, 2000, 5000]
            r_polish, pts_polish = _run_optimization(
                best_pts,
                time_limit=remaining_time * 0.4,
                betas=betas_polish
            )
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
            
            # Then do the active-set local search
            elapsed = time.time() - t0
            remaining_time = _TIME_LIMIT - 3 - elapsed
            if remaining_time > 5:
                r_active, pts_active = _active_set_local_search(
                    best_pts,
                    time_limit=remaining_time * 0.4
                )
                if r_active > best_r:
                    best_r = r_active
                    best_pts = pts_active
            
            # Then do the final high-precision polish
            elapsed = time.time() - t0
            remaining_time = _TIME_LIMIT - 3 - elapsed
            if remaining_time > 3:
                r_final, pts_final = _final_polish(
                    best_pts,
                    time_limit=remaining_time
                )
                if r_final > best_r:
                    best_r = r_final
                    best_pts = pts_final

    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))

    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
