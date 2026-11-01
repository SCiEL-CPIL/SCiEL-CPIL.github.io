"""Circle packing in perimeter-4 rectangle: SLSQP with constraints + contact-graph polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _force_relax(z, max_iter=800):
    """Force-directed relaxation to resolve initial overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-8:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]
        else:
            break

        # Adaptive step
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _slsqp_with_constraints(z0, time_budget):
    """SLSQP optimization with explicit inequality constraints."""
    n = N

    def obj(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    # Build constraints
    constraints = []

    # Boundary constraints: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]  # x_i - r_i >= 0
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]  # w - x_i - r_i >= 0
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]  # y_i - r_i >= 0
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]  # h - y_i - r_i >= 0
        })

    # Pairwise non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3 * i] - p[3 * j]) ** 2 +
                    (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                    (p[3 * i + 2] + p[3 * j + 2]) ** 2
                )
            })

    maxiter = int(min(1000, max(200, time_budget * 80)))
    try:
        res = minimize(obj, z0, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        return res.x
    except Exception:
        return None


def _penalty_objective(z, penalty_weight=5000.0):
    """Penalty-based objective for SLSQP when constraints are too many."""
    n = N
    x = z[0::3][:n]
    y = z[1::3][:n]
    r = z[2::3][:n]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(n, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _slsqp_penalty(z0, time_budget, penalty_weight=5000.0):
    """SLSQP with penalty-based objective."""
    n = N

    def obj(zv):
        return _penalty_objective(zv, penalty_weight)

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]
    maxiter = int(min(800, max(150, time_budget * 60)))
    try:
        res = minimize(obj, z0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        return res.x
    except Exception:
        return None


def _contact_graph_polish(z, time_budget):
    """
    Contact-graph based optimization: identify circles that are "stuck" (limited by
    contacts with neighbors or walls), then optimize their positions with tight constraints.
    """
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Identify contact graph: circles that are in contact (gap < threshold)
    contact_thresh = 0.01
    contact_graph = {i: set() for i in range(n)}

    for i in range(n):
        # Check wall contacts
        if x[i] - r[i] < contact_thresh:
            contact_graph[i].add('left')
        if w - x[i] - r[i] < contact_thresh:
            contact_graph[i].add('right')
        if y[i] - r[i] < contact_thresh:
            contact_graph[i].add('bottom')
        if h - y[i] - r[i] < contact_thresh:
            contact_graph[i].add('top')

        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            gap = d - (r[i] + r[j])
            if gap < contact_thresh:
                contact_graph[i].add(j)
                contact_graph[j].add(i)

    # Find circles with the most constraints (most contacts)
    constraint_count = []
    for i in range(n):
        count = len(contact_graph[i])
        constraint_count.append(count)

    # Sort by constraint count (most constrained first)
    order = sorted(range(n), key=lambda i: -constraint_count[i])

    # For each circle in order, try to reposition and expand
    improved = True
    max_rounds = 10
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False

        for i in order:
            if time.time() > time_budget:
                break

            # Try to expand radius i to max possible
            r_max = _max_radius_for_circle(x, y, r, w, h, i)

            if r_max > r[i] + 1e-7:
                # Binary search for true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-8:
                    r[i] = new_r
                    improved = True
            else:
                # Try to reposition to allow larger radius
                # Find the most constraining neighbor/wall
                min_gap = float('inf')
                best_dx, best_dy = 0.0, 0.0

                # Check walls
                d_left = x[i] - r[i]
                d_right = w - x[i] - r[i]
                d_bottom = y[i] - r[i]
                d_top = h - y[i] - r[i]

                if d_left < min_gap:
                    min_gap = d_left
                    best_dx, best_dy = 1.0, 0.0
                if d_right < min_gap:
                    min_gap = d_right
                    best_dx, best_dy = -1.0, 0.0
                if d_bottom < min_gap:
                    min_gap = d_bottom
                    best_dx, best_dy = 0.0, 1.0
                if d_top < min_gap:
                    min_gap = d_top
                    best_dx, best_dy = 0.0, -1.0

                # Check other circles
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        if d > 1e-6:
                            best_dx = dx / d
                            best_dy = dy / d

                if min_gap < 0.03:
                    # Try small moves away from constraint
                    found = False
                    for step in [0.01, 0.005, 0.003, 0.002, 0.001, 0.0005]:
                        new_x = x[i] + best_dx * step
                        new_y = y[i] + best_dy * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i], y[i] = new_x, new_y
                            found = True
                            # Try to expand after moving
                            r_max2 = _max_radius_for_circle(x, y, r, w, h, i)
                            if r_max2 > r[i] + 1e-7:
                                lo = r[i]
                                hi = r_max2
                                for _ in range(25):
                                    mid = (lo + hi) / 2.0
                                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                                        lo = mid
                                    else:
                                        hi = mid
                                new_r = lo
                                if new_r > r[i] + 1e-8:
                                    r[i] = new_r
                                    improved = True
                            break

        if not improved:
            break

    # Final uniform expansion
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _tight_polish(z, time_budget):
    """Tight final polish with force-directed relaxation and per-circle expansion."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Force-directed relaxation
    dt = 0.0005
    max_iter = int(min(5000, time_budget * 200))
    for it in range(max_iter):
        if time.time() > time_budget - 5:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-10)
        if max_force > 0.05:
            dt = 0.0002
        elif max_force > 0.01:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Per-circle expansion
    for pass_num in range(10):
        if time.time() > time_budget - 2:
            break
        for i in range(n):
            if time.time() > time_budget - 2:
                break
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-8:
                lo = r[i]
                hi = r_max
                for _ in range(35):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-9:
                    r[i] = new_r

    # Final uniform expansion with adaptive factors
    for _ in range(1000):
        if time.time() > time_budget - 1:
            break
        for factor in [1.0005, 1.0003, 1.0002, 1.0001, 1.00005, 1.00001]:
            r_trial = r * factor
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
                break
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _get_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    side = int(np.ceil(np.sqrt(N)))

    # Structured grid with various aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55,
                     2.0, 0.5, 1.2, 0.8, 0.9, 1.1, 0.7, 1.4, 0.65, 1.25,
                     1.6, 0.625, 1.8, 0.56, 1.1, 0.91]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.45
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    # Hexagonal-like packing for some aspect ratios
    for w0 in [1.0, 1.2, 0.8, 1.4, 0.7]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        # Hexagonal arrangement
        row_spacing = 0.866  # sqrt(3)/2
        col_spacing = 1.0
        for i in range(N):
            row, col = divmod(i, 5)
            cx = (col + 0.5 * (row % 2)) * col_spacing * (w0 / 5.0)
            cy = (row + 0.5) * row_spacing * (h0 / 4.0)
            cx = np.clip(cx, 0.05, w0 - 0.05)
            cy = np.clip(cy, 0.05, h0 - 0.05)
            r0 = min(w0 / 5.0, h0 / 4.0) * 0.4
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(15):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.05, 0.12)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    return guesses


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    guesses = _get_initial_guesses()

    # Phase 1: Multi-start optimization
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 40:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=500)

        # Try penalty-based SLSQP (faster, fewer constraints)
        remaining = deadline - (time.time() - start_time) - 30
        if remaining > 10:
            z_opt = _slsqp_penalty(z0_relaxed, remaining * 0.5)
            if z_opt is not None:
                valid, val = check_validity(z_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()

        # Try constraint-based SLSQP (more accurate, slower)
        remaining = deadline - (time.time() - start_time) - 25
        if remaining > 15 and best_result is not None:
            # Use the best result so far as a better starting point
            z_opt2 = _slsqp_with_constraints(z0_relaxed, remaining * 0.4)
            if z_opt2 is not None:
                valid, val = check_validity(z_opt2)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt2.copy()

    # Phase 2: Multi-start from perturbed best
    if best_result is not None and time.time() < deadline - 50:
        rng2 = np.random.default_rng(123)
        for attempt in range(8):
            if time.time() > deadline - 45:
                break
            z_pert = best_result.copy()
            noise_scale = 0.003 * (attempt + 1)
            for k in range(3 * N):
                z_pert[k] += rng2.normal(0, noise_scale)
            z_pert[-1] = np.clip(z_pert[-1], 0.4, 1.6)
            r_idx = np.arange(2, 3 * N, 3)
            z_pert[r_idx] = np.maximum(z_pert[r_idx], 0.01)

            remaining = deadline - (time.time() - start_time) - 30
            if remaining > 10:
                z_opt = _slsqp_penalty(z_pert, remaining * 0.4)
                if z_opt is not None:
                    valid, val = check_validity(z_opt)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_opt.copy()

                # Also try constraint-based
                remaining2 = deadline - (time.time() - start_time) - 25
                if remaining2 > 12:
                    z_opt2 = _slsqp_with_constraints(z_pert, remaining2 * 0.3)
                    if z_opt2 is not None:
                        valid, val = check_validity(z_opt2)
                        if valid and val > best_value:
                            best_value = val
                            best_result = z_opt2.copy()

    # Phase 3: Contact-graph polish
    if best_result is not None and time.time() < deadline - 25:
        contact_opt = _contact_graph_polish(best_result, deadline - (time.time() - start_time) - 15)
        if contact_opt is not None:
            valid, val = check_validity(contact_opt)
            if valid and val > best_value:
                best_value = val
                best_result = contact_opt

    # Phase 4: Tight final polish
    if best_result is not None and time.time() < deadline - 15:
        polished = _tight_polish(best_result, deadline - (time.time() - start_time) - 5)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # Phase 5: Final micro-polish: per-circle radius maximization
    if best_result is not None and time.time() < deadline - 2:
        x_f = best_result[0::3][:N].copy()
        y_f = best_result[1::3][:N].copy()
        r_f = best_result[2::3][:N].copy()
        w_f = best_result[-1]
        h_f = 2.0 - w_f

        for i in range(N):
            if time.time() > deadline - 1:
                break
            r_max = _max_radius_for_circle(x_f, y_f, r_f, w_f, h_f, i)
            if r_max > r_f[i] + 1e-10:
                lo = r_f[i]
                hi = r_max
                for _ in range(35):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x_f, y_f, np.where(np.arange(N) == i, mid, r_f), w_f, h_f, i, x_f[i], y_f[i]):
                        lo = mid
                    else:
                        hi = mid
                r_f[i] = lo

        if _is_valid(x_f, y_f, r_f, w_f, h_f):
            z_final = np.empty(3 * N + 1)
            z_final[0::3][:N] = x_f
            z_final[1::3][:N] = y_f
            z_final[2::3][:N] = r_f
            z_final[-1] = w_f
            valid, val = check_validity(z_final)
            if valid and val > best_value:
                best_value = val
                best_result = z_final

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
