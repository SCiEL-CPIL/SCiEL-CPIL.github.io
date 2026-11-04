"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 26

# Deadline guard: 145s wall-clock (5s safety margin under 150s)
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START) - 3.0


def _kinetic_annealing(pos, deadline, n_steps=20000, seed=0):
    """
    Kinetic annealing: treat circles as rigid bodies with mass ~ r^2.
    Apply repulsive collision impulses + gentle centering force.
    Symplectic (velocity-Verlet) integrator with adaptive damping.
    Returns best position array found.
    """
    rng = np.random.RandomState(seed)
    n = len(pos)
    pos_arr = pos.copy().astype(np.float64)
    vel = np.zeros_like(pos_arr)
    
    # Initial radii estimate for mass
    r0 = _max_radius_for_positions(pos_arr)
    r0 = np.maximum(r0, 0.01)
    mass = r0 ** 2
    mass = np.maximum(mass, 0.001)
    
    best_pos = pos_arr.copy()
    best_obj = _objective(pos_arr)
    
    dt = 0.002
    damping = 0.985
    repulse_strength = 0.5
    center_pull = 0.001
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Progress-based annealing: reduce repulsion, increase damping
        progress = step / n_steps
        rep = repulse_strength * (1.0 - 0.7 * progress)
        damp = damping + 0.01 * progress
        cpull = center_pull * (0.5 + progress)
        
        # Compute forces
        forces = np.zeros_like(pos_arr)
        
        # Pairwise repulsion (only for close pairs to save time)
        dists = cdist(pos_arr, pos_arr)
        np.fill_diagonal(dists, np.inf)
        r_est = _max_radius_for_positions(pos_arr)
        
        # Vectorized pairwise forces
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = pos_arr[i_idx, 0] - pos_arr[j_idx, 0]
        dy = pos_arr[i_idx, 1] - pos_arr[j_idx, 1]
        d = np.sqrt(dx**2 + dy**2)
        safe_d = np.where(d < 1e-10, 1e-10, d)
        rs_sum = r_est[i_idx] + r_est[j_idx]
        
        # Repulsion: strong when overlapping, moderate when close
        overlap = np.maximum(0, rs_sum - d)
        close = np.maximum(0, rs_sum * 1.5 - d)
        force_mag = rep * (overlap * 5.0 + close * 0.3)
        
        fx = force_mag * dx / safe_d
        fy = force_mag * dy / safe_d
        
        np.add.at(forces, i_idx, np.column_stack([-fx, -fy]))
        np.add.at(forces, j_idx, np.column_stack([fx, fy]))
        
        # Centering force (gentle pull toward [0.5, 0.5])
        center = np.array([0.5, 0.5])
        center_force = cpull * (center - pos_arr)
        forces += center_force
        
        # Acceleration (mass-weighted)
        acc = forces / mass[:, np.newaxis]
        
        # Velocity-Verlet step
        vel = vel * damp + acc * dt
        pos_arr = pos_arr + vel * dt
        
        # Clip to [0, 1] and zero velocity at boundaries
        for dim in range(2):
            at_lo = pos_arr[:, dim] < 0
            at_hi = pos_arr[:, dim] > 1
            pos_arr[at_lo, dim] = 0.0
            pos_arr[at_hi, dim] = 1.0
            vel[at_lo, dim] = 0.0
            vel[at_hi, dim] = 0.0
        
        # Evaluate and track best
        obj = _objective(pos_arr)
        if obj > best_obj:
            best_obj = obj
            best_pos = pos_arr.copy()
    
    return best_pos, best_obj


def _max_radius_for_positions(pos):
    """Vectorized: given positions (N,2), compute max radius for each circle."""
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


def _soft_objective(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad). Vectorized soft-penalty formulation.
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    """
    n = len(x) // 3
    xs, ys, rs = x[0::3], x[1::3], x[2::3]

    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    energy = obj + mu * (bpen + open_)

    grad = np.zeros_like(x)
    grad[2::3] = -1.0

    # Boundary penalty gradient
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))
    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist
        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

    return energy, grad


def _soft_grad(x, n=N, mu=1000.0):
    return _soft_objective(x, n, mu)[1]


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


def _project_feasible(pos, max_iter=400, tol=1e-10):
    """
    Project a position array to feasibility by iteratively resolving violations.
    Reduces radii and adjusts positions to eliminate boundary/overlap violations.
    Returns a feasible position array (radii recomputed from positions).
    """
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        xs, ys = pos[:, 0], pos[:, 1]
        radii = _max_radius_for_positions(pos)

        # Boundary: ensure radius <= min distance to boundary
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
        radii = np.minimum(radii, np.maximum(min_d, 0.0))

        # Overlap: reduce radii to resolve overlaps (vectorized)
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = radii[i_idx] + radii[j_idx]
        ov = np.maximum(0, rs_sum - dist)

        if not np.any(ov > tol):
            return _max_radius_for_positions(pos)

        # Reduce radii to resolve overlaps (vectorized accumulation)
        reduction = np.zeros(n)
        np.add.at(reduction, i_idx, ov / 2.0 + tol)
        np.add.at(reduction, j_idx, ov / 2.0 + tol)
        radii = np.maximum(radii - reduction, 0.005)

        # Enforce radius <= boundary
        radii = np.minimum(radii, np.maximum(min_d, 0.0))
        radii = np.maximum(radii, 0.0)

    return _max_radius_for_positions(pos)


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


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess, scaled to fit the unit square."""
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est

    xs, ys = [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        start_x = 0.5 - total_width / 2
        for j in range(count):
            x = start_x + j * x_spacing
            if i % 2 == 1:
                x += x_spacing / 2
            xs.append(x)
            ys.append(y)

    xs_arr = np.array(xs[:n])
    ys_arr = np.array(ys[:n])
    min_x = np.min(xs_arr)
    max_x = np.max(xs_arr)
    min_y = np.min(ys_arr)
    max_y = np.max(ys_arr)
    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height)
    xs_arr = (xs_arr - min_x) * scale
    ys_arr = (ys_arr - min_y) * scale
    xs_arr += (1.0 - width * scale) / 2
    ys_arr += (1.0 - height * scale) / 2
    return np.column_stack([xs_arr, ys_arr])


def _simulated_annealing(pos, t_start=0.5, t_end=0.0005, n_iter=40000, seed=42, deadline=None):
    """Simulated annealing to maximize sum of radii."""
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


def _best_fit_local_search(pos, n_iter=8000, seed=42, deadline=None):
    """Best-fit local search with targeted moves away from binding constraints."""
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved_any = False
        order = rng.permutation(n)

        for i in order:
            if deadline is not None and time.time() > deadline:
                break

            x_i, y_i = current_pos[i]
            r_i = _max_radius_for_positions(current_pos)[i]

            boundary_dist = min(x_i, 1.0 - x_i, y_i, 1.0 - y_i)
            if boundary_dist <= r_i + 1e-12:
                candidates = []
                if x_i <= 1.0 - x_i and x_i <= y_i and x_i <= 1.0 - y_i:
                    candidates.append((x_i + 0.01, y_i))
                    candidates.append((x_i + 0.005, y_i + 0.005))
                    candidates.append((x_i + 0.005, y_i - 0.005))
                elif 1.0 - x_i <= x_i and 1.0 - x_i <= y_i and 1.0 - x_i <= 1.0 - y_i:
                    candidates.append((x_i - 0.01, y_i))
                    candidates.append((x_i - 0.005, y_i + 0.005))
                    candidates.append((x_i - 0.005, y_i - 0.005))
                elif y_i <= x_i and y_i <= 1.0 - x_i and y_i <= 1.0 - y_i:
                    candidates.append((x_i, y_i + 0.01))
                    candidates.append((x_i + 0.005, y_i + 0.005))
                    candidates.append((x_i - 0.005, y_i + 0.005))
                else:
                    candidates.append((x_i, y_i - 0.01))
                    candidates.append((x_i + 0.005, y_i - 0.005))
                    candidates.append((x_i - 0.005, y_i - 0.005))
            else:
                d = cdist(current_pos, current_pos[i:i+1]).flatten()
                d[i] = np.inf
                j = int(np.argmin(d))
                if d[j] > 1e-15:
                    dx = (x_i - current_pos[j, 0]) / d[j]
                    dy = (y_i - current_pos[j, 1]) / d[j]
                    step = 0.008
                    candidates = [
                        (x_i + dx * step, y_i + dy * step),
                        (x_i + dx * step * 0.5, y_i + dy * step * 0.5),
                        (x_i + dx * step * 0.7 + dy * 0.004, y_i + dy * step * 0.7 - dx * 0.004),
                        (x_i + dx * step * 0.7 - dy * 0.004, y_i + dy * step * 0.7 + dx * 0.004),
                    ]
                else:
                    candidates = [(x_i + 0.01, y_i), (x_i - 0.01, y_i),
                                  (x_i, y_i + 0.01), (x_i, y_i - 0.01)]

            for _ in range(3):
                candidates.append((x_i + rng.uniform(-0.015, 0.015),
                                   y_i + rng.uniform(-0.015, 0.015)))

            best_new_pos = None
            best_new_obj = current_obj

            for cx, cy in candidates:
                cx = np.clip(cx, 0.0, 1.0)
                cy = np.clip(cy, 0.0, 1.0)
                new_pos = current_pos.copy()
                new_pos[i] = [cx, cy]
                new_obj = _objective(new_pos)
                if new_obj > best_new_obj + 1e-14:
                    best_new_obj = new_obj
                    best_new_pos = new_pos

            if best_new_pos is not None:
                current_pos = best_new_pos
                current_obj = best_new_obj
                improved_any = True

        if not improved_any:
            for _ in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                i = rng.randint(n)
                x_i, y_i = current_pos[i]
                dx = rng.uniform(-0.003, 0.003)
                dy = rng.uniform(-0.003, 0.003)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(x_i + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(y_i + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-14:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved_any = True
            if not improved_any:
                break

    return current_pos, current_obj


def _local_search(pos, n_iter=4000, step_size=0.008, seed=42, deadline=None):
    """Local search: iteratively nudge each circle to improve the objective."""
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


def _gradient_polish(pos, deadline, beta=1000.0, maxiter=1500):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with the vectorized
    soft-penalty objective. Returns (pos, radii, obj) if improved and valid.
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
            _soft_objective, v0, args=(n, beta),
            jac=_soft_grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
        )
    except Exception:
        return pos, r_init, float(np.sum(r_init))

    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))

    v_opt = res.x
    x_opt = v_opt[0::3]
    y_opt = v_opt[1::3]
    pos_opt = np.column_stack([x_opt, y_opt])
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    if _is_valid(pos_opt, eps=1e-8):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def _grow_radii(pos, deadline, max_iter=500):
    """
    Post-optimization radius growth: iteratively try to increase each circle's
    radius by pushing neighbors slightly apart. Returns improved (pos, radii, obj).
    """
    n = len(pos)
    pos = pos.copy()
    radii = _max_radius_for_positions(pos)
    obj = float(np.sum(radii))

    for it in range(max_iter):
        if time.time() > deadline:
            break

        improved_any = False
        for i in range(n):
            if time.time() > deadline:
                break

            x_i, y_i = pos[i]
            r_i = radii[i]

            # Compute current limiting constraint
            d_left = x_i
            d_bot = y_i
            d_right = 1.0 - x_i
            d_top = 1.0 - y_i
            boundary_min = min(d_left, d_bot, d_right, d_top)

            # Nearest neighbor distance
            dists = np.sqrt(np.sum((pos - pos[i])**2, axis=1))
            dists[i] = np.inf
            j = int(np.argmin(dists))
            nn_dist = dists[j]
            nn_r = radii[j]
            pair_limit = (nn_dist - nn_r)

            # Current radius is limited by min(boundary, pair_limit)
            limit = min(boundary_min, pair_limit)
            if r_i >= limit - 1e-10:
                continue

            # Try to grow radius: move circle toward center of its "free space"
            # Compute direction that maximizes free space
            grow_delta = 0.001
            best_new_pos = pos.copy()
            best_new_obj = obj

            # Try small moves in 8 directions
            for dx in [-grow_delta, 0, grow_delta]:
                for dy in [-grow_delta, 0, grow_delta]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = np.clip(x_i + dx, 0.0, 1.0)
                    new_y = np.clip(y_i + dy, 0.0, 1.0)
                    new_pos = pos.copy()
                    new_pos[i] = [new_x, new_y]
                    new_radii = _max_radius_for_positions(new_pos)
                    new_obj = float(np.sum(new_radii))
                    if new_obj > best_new_obj + 1e-12:
                        best_new_obj = new_obj
                        best_new_pos = new_pos

            if best_new_pos is not pos:
                pos = best_new_pos
                radii = _max_radius_for_positions(pos)
                obj = best_new_obj
                improved_any = True

        if not improved_any:
            break

    return pos, radii, obj


def _high_precision_polish(pos, deadline, beta=2000.0, maxiter=2500):
    """
    High-precision final polish: project to feasibility first, then run
    gradient-based optimization with tighter tolerances and higher beta.
    """
    n = len(pos)

    # Project to feasibility to give the optimizer a clean start
    r_proj = _project_feasible(pos)
    pos_proj = pos.copy()

    r_init = _max_radius_for_positions(pos_proj)
    v0 = np.concatenate([pos_proj[:, 0], pos_proj[:, 1], r_init])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))

    try:
        res = minimize(
            _soft_objective, v0, args=(n, beta),
            jac=_soft_grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-17, 'gtol': 1e-13, 'maxls': 60}
        )
    except Exception:
        return pos, r_init, float(np.sum(r_init))

    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))

    v_opt = res.x
    x_opt = v_opt[0::3]
    y_opt = v_opt[1::3]
    pos_opt = np.column_stack([x_opt, y_opt])
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    if _is_valid(pos_opt, eps=1e-8):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Diverse initializations: grid + hex + random
    initializations = [_initial_grid(), _hex_packing_init()]
    for s in range(9):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 40:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=25000, seed=init_idx * 7 + 1,
            deadline=deadline - 30
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 25:
            break

        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=4000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 18
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        if time.time() > deadline - 15:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=1500, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 12
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Kinetic annealing from best solution (new exploration strategy)
    if time.time() < deadline - 20:
        ka_pos, ka_obj = _kinetic_annealing(
            best_pos, deadline=deadline - 15,
            n_steps=15000, seed=42
        )
        if ka_obj > best_obj:
            best_pos = ka_pos
            best_obj = ka_obj

    # Phase 3: Best-fit local search on best solution
    if time.time() < deadline - 15:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=5000,
            seed=777, deadline=deadline - 12
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 4: Standard local search with finer steps
    if time.time() < deadline - 10:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=3000, step_size=0.002,
            seed=999, deadline=deadline - 8
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 5: Gradient-based polish (vectorized soft surrogate)
    if time.time() < deadline - 6:
        polished = _gradient_polish(best_pos, deadline - 3)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 6: Radius growth polish (post-optimization expansion)
    if time.time() < deadline - 4:
        grown_pos, grown_r, grown_obj = _grow_radii(best_pos, deadline - 2)
        if grown_obj > best_obj:
            best_pos = grown_pos
            best_obj = grown_obj

    # Phase 7: High-precision final polish (project + tight gradient)
    if time.time() < deadline - 2:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 1)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
