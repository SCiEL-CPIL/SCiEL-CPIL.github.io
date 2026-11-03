"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)


def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))


def _best_fit_local_search(pos, n_iter=8000, seed=42, deadline=None):
    """
    Dense grid-based best-fit local search: for each circle, evaluate a dense
    grid of candidate positions around the current position and pick the best.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)

    # Grid offsets: 11x11 = 121 candidates per circle
    grid_range = np.linspace(-0.015, 0.015, 11)
    offsets = np.array([(dx, dy) for dx in grid_range for dy in grid_range])

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved_any = False
        order = rng.permutation(n)

        for i in order:
            if deadline is not None and time.time() > deadline:
                break

            x_i, y_i = current_pos[i]
            # Generate candidate positions
            cand_x = np.clip(x_i + offsets[:, 0], 0.0, 1.0)
            cand_y = np.clip(y_i + offsets[:, 1], 0.0, 1.0)

            best_new_obj = current_obj
            best_idx = -1

            for ci in range(len(cand_x)):
                new_pos = current_pos.copy()
                new_pos[i] = [cand_x[ci], cand_y[ci]]
                new_obj = _objective(new_pos)
                if new_obj > best_new_obj + 1e-14:
                    best_new_obj = new_obj
                    best_idx = ci

            if best_idx >= 0:
                current_pos[i] = [cand_x[best_idx], cand_y[best_idx]]
                current_obj = best_new_obj
                improved_any = True

        if not improved_any:
            # Try with smaller random steps
            for _ in range(30):
                if deadline is not None and time.time() > deadline:
                    break
                i = rng.randint(n)
                dx = rng.uniform(-0.004, 0.004)
                dy = rng.uniform(-0.004, 0.004)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-14:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved_any = True
            if not improved_any:
                break

    return current_pos, current_obj


def _is_valid(pos, eps=1e-9):
    """Check pairwise non-overlap and boundary containment with computed radii."""
    radii = _max_radius_for_positions(pos)
    dists = cdist(pos, pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dists[i, j] < radii[i] + radii[j] - eps:
                return False
        if (pos[i, 0] < radii[i] - eps or pos[i, 0] > 1.0 - radii[i] + eps or
                pos[i, 1] < radii[i] - eps or pos[i, 1] > 1.0 - radii[i] + eps):
            return False
    return True


def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])


def _initial_random(seed=0):
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 5000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > 0.06 for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)


def _simulated_annealing(pos, t_start=0.5, t_end=0.0005, n_iter=40000, seed=42, deadline=None):
    """
    Simulated annealing to maximize sum of radii.
    Uses vectorized _max_radius_for_positions for speed.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    best_pos = current_pos.copy()
    best_obj = current_obj

    n = len(current_pos)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        t = t_start * (t_end / t_start) ** (it / n_iter)

        i = rng.randint(n)
        step_size = 0.04 * t + 0.0008
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)

        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

        new_obj = _objective(new_pos)
        delta = new_obj - current_obj

        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_pos = current_pos.copy()
                best_obj = current_obj

    return best_pos, best_obj


def _local_search(pos, n_iter=4000, step_size=0.008, seed=42, deadline=None):
    """
    Local search: iteratively nudge each circle to improve the objective.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    n = len(current_pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        it += 1
        order = rng.permutation(n)
        for i in order:
            for _ in range(15):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-13:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved = True
                    break

    return current_pos, current_obj


def _high_precision_polish(pos, deadline):
    """
    High-precision final polish: run gradient-based optimization with progressive
    beta values to smooth the landscape initially and sharpen it later.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))
    
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in betas:
        if time.time() > deadline - 1:
            break
            
        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])

        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 0.5))

        try:
            res = minimize(
                _smooth_objective, v0, args=(n, beta),
                jac=_smooth_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 60}
            )
        except Exception:
            continue

        if time.time() > deadline - 1:
            break

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2 * n]
        pos_opt = np.column_stack([x_opt, y_opt])
        
        # Clamp to ensure validity before checking
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        if _is_valid(pos_opt, eps=1e-8) and obj_opt > best_obj:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()

    return best_pos, best_r, best_obj


def _smooth_objective(v, n, beta=40.0):
    """
    Smooth objective for gradient-based polish.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Minimize: -sum(r) + penalty * (smooth wall violations + smooth pairwise violations)
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    # Wall violations: r_i - min(x_i, 1-x_i, y_i, 1-y_i)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Pairwise violations: r_i + r_j - dist_ij
    total_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_pen += np.logaddexp(0.0, beta * viol) / beta

    return -np.sum(r) + 50.0 * (wall_pen + total_pen)


def _smooth_gradient(v, n, beta=40.0):
    """
    Vectorized analytic gradient of _smooth_objective.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)

    # --- Wall penalty (vectorized) ---
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    # sigmoid(beta * wall_viol)
    z = beta * wall_viol
    sig = np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    # d/dwall_viol of log(1+exp(beta*viol))/beta = sig
    # dwall_viol/d r_i = 1; dwall_viol/d(x_i or y_i) = -1 if that side is the min
    gr += 50.0 * sig
    # Determine which wall is binding for each circle
    wx = np.minimum(x, 1.0 - x)
    wy = np.minimum(y, 1.0 - y)
    # wall_min = min(wx, wy)
    # If wx <= wy, then wall_min = wx, and dwall_min/dx = -1 if x<=0.5 else +1
    use_x = wx <= wy
    # For x-component:
    dx_sign = np.where(x <= 0.5, -1.0, 1.0)  # d(x or 1-x)/dx
    gx -= 50.0 * sig * use_x * dx_sign
    # For y-component:
    dy_sign = np.where(y <= 0.5, -1.0, 1.0)
    gy -= 50.0 * sig * (~use_x) * dy_sign

    # --- Pairwise penalty (vectorized) ---
    # Build (n, n) distance matrix
    dx_mat = x[:, None] - x[None, :]
    dy_mat = y[:, None] - y[None, :]
    dist_mat = np.sqrt(dx_mat * dx_mat + dy_mat * dy_mat)
    np.fill_diagonal(dist_mat, np.inf)

    viol_mat = r[:, None] + r[None, :] - dist_mat
    np.fill_diagonal(viol_mat, 0.0)

    z_mat = beta * viol_mat
    # sigmoid
    sig_mat = np.where(z_mat > 0, 1.0 / (1.0 + np.exp(-z_mat)), np.exp(z_mat) / (1.0 + np.exp(z_mat)))
    np.fill_diagonal(sig_mat, 0.0)

    factor_mat = 50.0 * sig_mat  # (n, n)

    # grad_x[i] += sum_j factor[i,j] * (x_i - x_j) / dist[i,j]
    #          -= sum_j factor[j,i] * (x_j - x_i) / dist[j,i]  = -sum_j factor[j,i] * (x_j-x_i)/dist
    # Since factor is symmetric and dist is symmetric, this is:
    # grad_x[i] = sum_j factor[i,j] * dx_mat[i,j] / dist_mat[i,j]
    # (because the second term with j->i swap gives -factor[i,j]*(x_j-x_i)/dist = factor[i,j]*(x_i-x_j)/dist)
    # Wait, let me re-derive. The pairwise sum is over i<j:
    # f_ij = 50 * sig(r_i + r_j - dist_ij)
    # df/dx_i = f_ij * (x_i - x_j)/dist_ij  (for the i<j term)
    # df/dx_j = -f_ij * (x_i - x_j)/dist_ij = f_ij * (x_j - x_i)/dist_ij
    # So for circle i, contributions from all j != i:
    #   if i < j: f_ij * (x_i - x_j)/dist_ij
    #   if i > j: f_ji * (x_j - x_i)/dist_ji = f_ij * (x_j - x_i)/dist_ij (since f is symmetric)
    # Total for x_i: sum_{j != i} f_ij * (x_i - x_j)/dist_ij  ... wait no.
    # For i > j: the term is f_ji * (x_j - x_i)/dist_ji. Since f_ji = f_ij and dist_ji = dist_ij:
    #   = f_ij * (x_j - x_i)/dist_ij = -f_ij * (x_i - x_j)/dist_ij
    # So total: sum_{j<i} (-f_ij*(x_i-x_j)/dist_ij) + sum_{j>i} (f_ij*(x_i-x_j)/dist_ij)
    # That's not simply sum_{j!=i} f_ij*(x_i-x_j)/dist_ij.
    # Let me just use the matrix approach directly:
    # grad_x[i] = sum_{j>i} f_ij*(x_i-x_j)/d_ij + sum_{j<i} f_ji*(x_j-x_i)/d_ji
    #           = sum_{j>i} f_ij*dx_ij/d_ij - sum_{j<i} f_ij*dx_ij/d_ij  (where dx_ij = x_i-x_j)
    # Hmm, let me just compute it as:
    # grad_x[i] = sum_j f_ij * (x_i - x_j) / d_ij  for j where i<j
    #           + sum_j f_ji * (x_j - x_i) / d_ji  for j where i>j
    #           = sum_j f_ij * (x_i - x_j) / d_ij  for j where i<j
    #           - sum_j f_ij * (x_i - x_j) / d_ij  for j where i>j
    # This is getting confusing. Let me use a cleaner formulation.
    # 
    # The objective has term: sum_{i<j} 50*softplus(beta*(r_i+r_j-d_ij))
    # d/dx_i of this = sum_{j>i} 50*sig_ij*(x_i-x_j)/d_ij - sum_{j<i} 50*sig_ji*(x_i-x_j)/d_ji
    #                 = sum_{j>i} f_ij*dx_ij/d_ij - sum_{j<i} f_ij*dx_ji/d_ji
    #                 = sum_{j>i} f_ij*dx_ij/d_ij + sum_{j<i} f_ij*dx_ij/d_ij  (since dx_ji = -dx_ij, d_ji=d_ij)
    # Wait: for j<i, the derivative w.r.t x_i of f_ji*softplus(...) where f_ji depends on x_i via d_ji:
    # d/dx_i [50*sig(beta*(r_j+r_i-d_ji))] = 50*sig_ji * beta * d/dx_i(r_j+r_i-d_ji) / beta
    # = 50*sig_ji * (-d/dx_i d_ji) = 50*sig_ji * (-(x_i-x_j)/d_ji) ... wait
    # d_ji = sqrt((x_j-x_i)^2 + (y_j-y_i)^2), d/dx_i d_ji = (x_j-x_i)/d_ji = -dx_ij/d_ij
    # So d/dx_i [softplus(beta*(r_j+r_i-d_ji))] = sig_ji * (-d/dx_i d_ji) = sig_ji * dx_ij/d_ij
    # Hmm wait: d/dx_i of (r_j + r_i - d_ji) = -d/dx_i d_ji = -(x_j-x_i)/d_ji = (x_i-x_j)/d_ji = dx_ij/d_ij
    # So grad contribution for j<i to x_i: 50 * sig_ji * dx_ij / d_ij
    # And for j>i: 50 * sig_ij * dx_ij / d_ij
    # So total: sum_{j!=i} 50 * sig_ij * dx_ij / d_ij  (using symmetry sig_ij = sig_ji)
    
    # Great! So:
    # grad_x[i] = sum_{j!=i} 50 * sig_ij * (x_i - x_j) / d_ij
    # grad_y[i] = sum_{j!=i} 50 * sig_ij * (y_i - y_j) / d_ij  
    # grad_r[i] = sum_{j!=i} 50 * sig_ij  (from pairwise) + 50*sig_wall_i (from wall) - 1

    ratio_mat = factor_mat / dist_mat  # (n,n), with 0 on diagonal
    gx += ratio_mat @ dx_mat  # Wait, this gives sum_j ratio[i,j]*dx[i,j] which is what we want
    # Actually: (ratio_mat @ dx_mat)[i] = sum_j ratio_mat[i,j] * dx_mat[i,j] = sum_j 50*sig_ij*dx_ij/d_ij ✓
    # But we need sum_j 50*sig_ij * dx_ij/d_ij, and ratio_mat[i,j]*dx_mat[i,j] = 50*sig_ij*dx_ij/d_ij ✓
    # However, matrix multiply (A@B)[i] = sum_j A[i,j]*B[j,i], not sum_j A[i,j]*B[i,j]!
    # So I need element-wise multiply then sum along axis=1.
    gx_pair = np.sum(factor_mat * dx_mat / dist_mat, axis=1)
    gy_pair = np.sum(factor_mat * dy_mat / dist_mat, axis=1)
    gr_pair = np.sum(factor_mat, axis=1)

    gx += gx_pair
    gy += gy_pair
    gr += gr_pair

    # Objective: -sum(r)
    gr -= 1.0

    grad = np.concatenate([gx, gy, gr])
    return grad


def _gradient_polish(pos, deadline, beta=40.0):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with smooth objective.
    Returns improved (pos, radii) if valid and better.
    """
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    v0 = np.concatenate([pos[:, 0], pos[:, 1], r_init])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
    for i in range(n):
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((0.0, 0.5))  # r

    try:
        res = minimize(
            _smooth_objective, v0, args=(n, beta),
            jac=_smooth_gradient, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 60}
        )
    except Exception:
        return pos, r_init

    if time.time() > deadline:
        return pos, r_init

    v_opt = res.x
    x_opt = v_opt[0:n]
    y_opt = v_opt[n:2 * n]

    pos_opt = np.column_stack([x_opt, y_opt])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    # Recompute exact radii from positions (the optimizer may have r > feasible)
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    # Verify validity
    if _is_valid(pos_opt, eps=1e-8):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid()]
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 30:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=30000, seed=init_idx * 7 + 1,
            deadline=deadline - 25
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 20:
            break

        # Best-fit local search (targeted moves)
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=5000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 15
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        # Additional standard local search pass
        if time.time() > deadline - 12:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=2000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 10
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 15:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=6000,
            seed=777, deadline=deadline - 12
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 10:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=4000, step_size=0.002,
            seed=999, deadline=deadline - 8
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Gradient-based polish (smooth surrogate) - multiple passes
    for pass_i in range(3):
        if time.time() < deadline - 8:
            polished = _gradient_polish(best_pos, deadline - 6)
            if len(polished) == 3:
                p_pol, r_pol, obj_pol = polished
                if obj_pol > best_obj:
                    best_pos = p_pol
                    best_obj = obj_pol
                else:
                    break

    # Phase 5: High-precision final polish (progressive beta)
    if time.time() < deadline - 5:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 2)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 6: Another gradient polish after high-precision
    if time.time() < deadline - 3:
        polished2 = _gradient_polish(best_pos, deadline - 1)
        if len(polished2) == 3:
            p_pol2, r_pol2, obj_pol2 = polished2
            if obj_pol2 > best_obj:
                best_pos = p_pol2
                best_obj = obj_pol2

    # Phase 7: Final fine-tuning local search
    if time.time() < deadline - 1:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=1000, step_size=0.001,
            seed=12345, deadline=deadline - 0.5
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result

# EVOLVE-BLOCK-END
