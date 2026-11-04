"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _soft_penalty(v, n, beta=30.0):
    """
    Soft penalty objective: minimize -sum(r) + penalty.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Returns (objective_value, total_radius, num_violations).
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Wall violations
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = np.maximum(0.0, r - wall_min)

    # Pairwise violations
    total_pair_viol = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            if viol > 0:
                total_pair_viol += viol
                n_pairs += 1

    wall_pen = np.sum(wall_viol)
    pair_pen = total_pair_viol

    obj = -np.sum(r) + 100.0 * (wall_pen + pair_pen)
    return obj, float(np.sum(r)), int(np.sum(wall_viol > 0) + n_pairs)


def _is_valid(xy, r, tol=1e-9):
    """Check if packing is valid (all circles inside square, disjoint interiors)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -tol:
            return False
        if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _hex_seed(n):
    """Generate a hexagonal close-packing seed for n circles in unit square."""
    rng = np.random.RandomState(42)
    # Start with a hexagonal grid pattern
    positions = []
    spacing = 0.18
    row = 0
    while len(positions) < n:
        offset = (row % 2) * spacing * 0.5
        col = 0
        while len(positions) < n:
            x = 0.1 + offset + col * spacing
            y = 0.1 + row * spacing * 0.866
            if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                positions.append([x, y])
            col += 1
        row += 1
    positions = positions[:n]
    xy = np.array(positions, dtype=float)
    r = np.full(n, 0.08)
    return xy, r


def _random_seed(n, seed=0):
    """Random seed with moderate radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.15, 0.85, size=(n, 2))
    r = rng.uniform(0.06, 0.12, size=n)
    return xy, r


def _repulsion_relax(xy, r, n_iters=200, deadline=None):
    """
    Relax circles via mutual repulsion + wall attraction to spread them out.
    This improves initial geometry before the main SA search.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        forces = np.zeros((n, 2))
        for i in range(n):
            # Wall repulsion (push away from walls)
            for axis in range(2):
                pos = xy[i, axis]
                rad = r[i]
                # Left wall
                if pos - rad < 0:
                    forces[i, axis] += 5.0 * (rad - pos)
                # Right wall
                if 1.0 - pos - rad < 0:
                    forces[i, axis] -= 5.0 * (1.0 - pos - rad)
                # Bottom wall
                if pos - rad < 0:
                    forces[i, axis] += 5.0 * (rad - pos)
                # Top wall
                if 1.0 - pos - rad < 0:
                    forces[i, axis] -= 5.0 * (1.0 - pos - rad)

            # Pairwise repulsion
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1e-12:
                    dist_sq = 1e-12
                dist = np.sqrt(dist_sq)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    f = 10.0 * overlap / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy

        # Apply forces with damping
        lr = 0.02 * (1.0 - it / n_iters) + 0.005
        xy += lr * forces
        # Clamp to square
        for i in range(n):
            xy[i, 0] = np.clip(xy[i, 0], r[i], 1.0 - r[i])
            xy[i, 1] = np.clip(xy[i, 1], r[i], 1.0 - r[i])
    return xy, r


def _simulated_annealing(xy, r, deadline, n_iter_max=5000, seed=0):
    """
    Simulated annealing search over (x, y, r) space.
    State: flat vector v of length 3*n.
    Move: Gaussian perturbation on a random subset of coordinates.
    Accept: Metropolis criterion on soft penalty objective.
    """
    n = len(xy)
    rng = np.random.RandomState(seed)

    v = np.concatenate([xy[:, 0], xy[:, 1], r])
    obj_cur, _, _ = _soft_penalty(v, n)

    best_v = v.copy()
    best_obj = obj_cur
    best_total = 0.0
    best_valid = False

    # Check if initial is valid
    if _is_valid(xy, r):
        best_total = float(np.sum(r))
        best_valid = True

    T_start = 0.5
    T_end = 0.001
    # Schedule: exponential cooling
    # We'll do n_iter_max steps, cooling from T_start to T_end
    cool_rate = (T_end / T_start) ** (1.0 / max(n_iter_max, 1))

    T = T_start
    step_size_base = 0.03

    for it in range(n_iter_max):
        if time.time() > deadline:
            break

        # Perturb: pick 1-3 random circles, perturb their x, y, and maybe r
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(n, size=n_perturb, replace=False)
        v_new = v.copy()
        for idx in indices:
            # Perturb position
            v_new[idx] += rng.normal(0, step_size_base)
            v_new[n + idx] += rng.normal(0, step_size_base)
            # Perturb radius (bias toward growth)
            r_perturb = rng.normal(0.005, step_size_base * 0.5)
            v_new[2*n + idx] += r_perturb
            # Clamp radius to [0.005, 0.45]
            v_new[2*n + idx] = np.clip(v_new[2*n + idx], 0.005, 0.45)
            # Clamp position to [0, 1]
            v_new[idx] = np.clip(v_new[idx], 0.0, 1.0)
            v_new[n + idx] = np.clip(v_new[n + idx], 0.0, 1.0)

        obj_new, total_new, _ = _soft_penalty(v_new, n)

        # Metropolis acceptance
        delta = obj_new - obj_cur
        if delta < 0 or rng.random() < np.exp(-delta / T):
            v = v_new
            obj_cur = obj_new

            # Track best
            if obj_new < best_obj:
                best_obj = obj_new
                best_v = v.copy()
                # Check validity of best
                bv_xy = np.column_stack([best_v[0:n], best_v[n:2*n]])
                bv_r = best_v[2*n:3*n]
                if _is_valid(bv_xy, bv_r):
                    best_total = float(np.sum(bv_r))
                    best_valid = True

        # Cool
        T *= cool_rate
        # Slightly decrease step size
        step_size_base *= 0.9995

    return best_v, best_total, best_valid


def _gradient_polish(xy, r, deadline, max_iter=300):
    """
    Short L-BFGS-B polish using a smooth softplus objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = [(0.001, 0.999)] * (2 * n) + [(0.001, 0.49)] * n

    beta = 40.0

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        x = v[0:n]
        y = v[n:2*n]
        r_ = v[2*n:3*n]
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r_ - wall_min
        wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)
        total_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = r_[i] + r_[j] - dist
                total_viol += np.logaddexp(0.0, beta * viol) / beta
        return -np.sum(r_) + 100.0 * (wall_pen + total_viol)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        x = v[0:n]
        y = v[n:2*n]
        r_ = v[2*n:3*n]
        g = np.zeros(3 * n)
        # Wall
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r_ - wall_min
        sig = 1.0 / (1.0 + np.exp(-beta * wall_viol))
        for i in range(n):
            s = sig[i]
            if s < 1e-12:
                continue
            dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
            mi = int(np.argmin(dists))
            g[2*n + i] += 100.0 * s
            if mi == 0:
                g[i] -= 100.0 * s
            elif mi == 1:
                g[i] += 100.0 * s
            elif mi == 2:
                g[n + i] -= 100.0 * s
            else:
                g[n + i] += 100.0 * s
        # Pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                if d_sq < 1e-20:
                    continue
                d = np.sqrt(d_sq)
                viol = r_[i] + r_[j] - d
                s = 1.0 / (1.0 + np.exp(-beta * viol))
                if s < 1e-12:
                    continue
                f = 100.0 * s
                g[i] += f * dx / d
                g[n + i] += f * dy / d
                g[j] -= f * dx / d
                g[n + j] -= f * dy / d
                g[2*n + i] += f
                g[2*n + j] += f
        g[2*n:3*n] -= 1.0
        return g

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10,
                                   'gtol': 1e-6, 'maxfun': 5000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        if _is_valid(new_xy, new_r):
            return new_xy, new_r
    except Exception:
        pass
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    best_total = 0.0
    best_valid = False
    best_xy = None
    best_r = None

    def _update_best(xy, r):
        nonlocal best_total, best_valid, best_xy, best_r
        if _is_valid(xy, r):
            total = float(np.sum(r))
            if total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()

    # Phase 1: Generate diverse seeds, relax, then SA on each
    seeds = []
    # Hex seed
    xy_h, r_h = _hex_seed(N)
    xy_h, r_h = _repulsion_relax(xy_h, r_h, n_iters=100, deadline=deadline)
    seeds.append((xy_h, r_h, 0))

    # Random seeds
    for s in range(6):
        if time.time() > deadline:
            break
        xy_s, r_s = _random_seed(N, seed=s)
        xy_s, r_s = _repulsion_relax(xy_s, r_s, n_iters=80, deadline=deadline)
        seeds.append((xy_s, r_s, 100 + s))

    # Run SA on each seed
    for idx, (xy_s, r_s, sa_seed) in enumerate(seeds):
        if time.time() > deadline:
            break
        # Check initial
        _update_best(xy_s, r_s)

        # SA search
        v_best, total_sa, valid_sa = _simulated_annealing(
            xy_s.copy(), r_s.copy(), deadline,
            n_iter_max=3000, seed=sa_seed)

        if valid_sa:
            v_xy = np.column_stack([v_best[0:N], v_best[N:2*N]])
            v_r = v_best[2*N:3*N]
            _update_best(v_xy, v_r)

        # Gradient polish from best
        if time.time() < deadline - 15:
            if best_valid and best_xy is not None:
                px, pr = _gradient_polish(best_xy.copy(), best_r.copy(), deadline, max_iter=200)
                _update_best(px, pr)

    # Phase 2: Additional SA restarts from best if time permits
    if best_valid and best_xy is not None:
        for extra in range(3):
            if time.time() > deadline - 20:
                break
            # Perturb best and run short SA
            rng = np.random.RandomState(200 + extra)
            xy_p = best_xy.copy() + rng.normal(0, 0.02, size=(N, 2))
            r_p = best_r.copy() + rng.normal(0.005, 0.01, size=N)
            r_p = np.clip(r_p, 0.01, 0.45)
            xy_p[:, 0] = np.clip(xy_p[:, 0], 0.0, 1.0)
            xy_p[:, 1] = np.clip(xy_p[:, 1], 0.0, 1.0)

            v_best, total_sa, valid_sa = _simulated_annealing(
                xy_p, r_p, deadline, n_iter_max=2000, seed=300 + extra)
            if valid_sa:
                v_xy = np.column_stack([v_best[0:N], v_best[N:2*N]])
                v_r = v_best[2*N:3*N]
                _update_best(v_xy, v_r)

            # Polish
            if time.time() < deadline - 10:
                px, pr = _gradient_polish(best_xy.copy(), best_r.copy(), deadline, max_iter=150)
                _update_best(px, pr)

    # Final polish
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        px, pr = _gradient_polish(best_xy.copy(), best_r.copy(), deadline, max_iter=400)
        _update_best(px, pr)

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final clipping and validation
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    if not _is_valid(final_xy, final_r):
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
