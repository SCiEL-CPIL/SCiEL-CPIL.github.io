import numpy as np
from scipy.optimize import minimize
import time

N = 21
_DEADLINE = 135.0  # safety margin before 150s kill
SOFTMIN_TAU = 0.005


def _compute_radii_vec(x, y, W, H):
    """Vectorized Voronoi radii computation."""
    n = len(x)
    # Wall distances
    r = np.minimum(x, W - x)
    r = np.minimum(r, y)
    r = np.minimum(r, H - y)
    # Pairwise distances (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    half_d = 0.5 * d
    r = np.minimum(r, half_d.min(axis=1))
    return r


def _softmin_radii_vec(x, y, W, H, tau=SOFTMIN_TAU):
    """Vectorized soft-min radii for differentiability."""
    n = len(x)
    # Wall distances
    d_left = x
    d_right = W - x
    d_bottom = y
    d_top = H - y

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    half_d = 0.5 * d

    # Stack all constraints: shape (n, 4 + (n-1))
    # For each i, constraints are [d_left[i], d_right[i], d_bottom[i], d_top[i], half_d[i,j] for j!=i]
    # Build full constraint matrix
    cons = np.zeros((n, n + 3))
    cons[:, 0] = d_left
    cons[:, 1] = d_right
    cons[:, 2] = d_bottom
    cons[:, 3] = d_top
    cons[:, 4:] = half_d
    # Set diagonal of half_d part to large value (self-distance)
    for i in range(n):
        cons[i, 4 + i] = 1e10

    # Soft-min along axis 1
    cons_clipped = np.clip(cons, 1e-8, None)
    log_sum = np.log(np.sum(np.exp(-cons_clipped / tau), axis=1))
    r = -tau * log_sum
    return r


def _sum_radii(x, y, W, H):
    r = _compute_radii_vec(x, y, W, H)
    return np.sum(r)


def _validate(x, y, r, W, H, tol=1e-6):
    """Check if circles are valid."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    if np.any(d < r[:, None] + r[None, :] - tol):
        return False
    return True


def _objective_soft(v, W, H, tau=SOFTMIN_TAU):
    """Soft-min objective for gradient-based optimization."""
    n = N
    x = v[:n]
    y = v[n:2*n]
    r = _softmin_radii_vec(x, y, W, H, tau)
    return -np.sum(r)


def _objective_exact(v, W, H):
    """Exact (nonsmooth) objective."""
    n = N
    x = v[:n]
    y = v[n:2*n]
    r = _compute_radii_vec(x, y, W, H)
    return -np.sum(r)


def _initial_grid(W, H, cols, rows, jitter=0.0, seed=None):
    """Grid initialization."""
    n = N
    xs = np.zeros(n)
    ys = np.zeros(n)
    k = 0
    rng = np.random.RandomState(seed) if seed is not None else np.random
    for i in range(cols):
        for j in range(rows):
            if k >= n:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            if jitter > 0:
                cx += (rng.rand() - 0.5) * W / cols * jitter
                cy += (rng.rand() - 0.5) * H / rows * jitter
            xs[k] = np.clip(cx, 0.01, W - 0.01)
            ys[k] = np.clip(cy, 0.01, H - 0.01)
            k += 1
    return xs, ys


def _initial_random(W, H, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    xs = rng.uniform(0.05, W - 0.05, N)
    ys = rng.uniform(0.05, H - 0.05, N)
    return xs, ys


def _sa_optimize(x, y, W, H, start_time, time_budget, rng, t0=0.3, t1=0.001, max_iter=50000):
    """Simulated annealing to optimize circle positions."""
    n = N
    cur_x = x.copy()
    cur_y = y.copy()
    cur_r = _compute_radii_vec(cur_x, cur_y, W, H)
    cur_val = np.sum(cur_r)

    best_x = cur_x.copy()
    best_y = cur_y.copy()
    best_val = cur_val

    step_size = 0.05
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        # Temperature cooling
        frac = it / max_iter
        t = t0 * (t1 / t0) ** frac

        # Pick random circle to move
        i = rng.randint(n)
        dx = (rng.rand() - 0.5) * 2 * step_size
        dy = (rng.rand() - 0.5) * 2 * step_size

        new_x = cur_x.copy()
        new_y = cur_y.copy()
        new_x[i] = np.clip(cur_x[i] + dx, 0.0, W)
        new_y[i] = np.clip(cur_y[i] + dy, 0.0, H)

        new_r = _compute_radii_vec(new_x, new_y, W, H)
        new_val = np.sum(new_r)

        delta = new_val - cur_val
        if delta > 0 or rng.rand() < np.exp(delta / t):
            cur_x, cur_y = new_x, new_y
            cur_r = new_r
            cur_val = new_val
            if cur_val > best_val:
                best_val = cur_val
                best_x = cur_x.copy()
                best_y = cur_y.copy()

        # Reduce step size slowly
        if it % 5000 == 0 and it > 0:
            step_size *= 0.7

    return best_x, best_y, best_val


def _gradient_polish(x, y, W, H, start_time, time_budget):
    """Polish using L-BFGS-B with soft-min objective."""
    n = N
    x0 = np.concatenate([x, y])
    bounds = [(0.0, W)] * n + [(0.0, H)] * n

    try:
        res = minimize(_objective_soft, x0, args=(W, H), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        x_new = np.clip(res.x[:n], 0.0, W)
        y_new = np.clip(res.x[n:2*n], 0.0, H)
    except Exception:
        return x, y, _sum_radii(x, y, W, H)

    r_new = _compute_radii_vec(x_new, y_new, W, H)
    val_new = np.sum(r_new)

    if _validate(x_new, y_new, r_new, W, H):
        # Try exact objective refinement
        try:
            res2 = minimize(_objective_exact, np.concatenate([x_new, y_new]),
                            args=(W, H), method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            x_pol = np.clip(res2.x[:n], 0.0, W)
            y_pol = np.clip(res2.x[n:2*n], 0.0, H)
            r_pol = _compute_radii_vec(x_pol, y_pol, W, H)
            val_pol = np.sum(r_pol)
            if _validate(x_pol, y_pol, r_pol, W, H) and val_pol > val_new:
                return x_pol, y_pol, val_pol
        except Exception:
            pass
        return x_new, y_new, val_new

    return x, y, _sum_radii(x, y, W, H)


def _optimize_for_aspect(W, H, start_x, start_y, start_time, time_budget, rng):
    """Full optimization pipeline for given aspect ratio."""
    # Phase 1: Simulated Annealing
    sa_x, sa_y, sa_val = _sa_optimize(start_x, start_y, W, H, start_time, time_budget, rng,
                                       t0=0.2, t1=0.001, max_iter=30000)

    # Phase 2: Gradient polish
    pol_x, pol_y, pol_val = _gradient_polish(sa_x, sa_y, W, H, start_time, time_budget)

    # Phase 3: Another SA round from polished result
    sa2_x, sa2_y, sa2_val = _sa_optimize(pol_x, pol_y, W, H, start_time, time_budget, rng,
                                          t0=0.1, t1=0.0005, max_iter=20000)

    # Phase 4: Final polish
    fin_x, fin_y, fin_val = _gradient_polish(sa2_x, sa2_y, W, H, start_time, time_budget)

    # Return best of all
    candidates = [(sa_x, sa_y, sa_val), (pol_x, pol_y, pol_val),
                  (sa2_x, sa2_y, sa2_val), (fin_x, fin_y, fin_val)]
    best = max(candidates, key=lambda c: c[2])
    return best[0], best[1], best[2]


def construct():
    start_time = time.time()
    time_budget = _DEADLINE - start_time - 5.0

    best_val = -1
    best_res = None

    rng = np.random.RandomState(42)

    # Aspect ratios to try (W + H = 2)
    # Square and near-square are typically best for dense packing
    aspect_ratios = [
        1.0, 1.05, 0.95, 1.1, 0.9, 1.15, 0.85, 1.2, 0.8, 1.25, 0.75,
        1.3, 0.7, 1.4, 0.65, 1.5, 0.6, 1.6, 0.55
    ]

    grid_configs = [
        (5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4),
        (6, 4), (4, 6), (5, 5), (3, 7), (7, 3), (5, 5), (4, 6),
        (6, 4), (5, 5), (4, 6), (6, 4), (5, 5)
    ]

    # Allocate time budget across aspects
    n_aspects = len(aspect_ratios)
    per_aspect_time = time_budget / n_aspects * 0.9  # 10% safety

    for ai, W in enumerate(aspect_ratios):
        H = 2.0 - W
        if H <= 0.1:
            continue

        # Check deadline
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break

        # Time for this aspect
        aspect_budget = min(per_aspect_time, time_budget - (time.time() - start_time))
        if aspect_budget < 2.0:
            break

        # Two restarts per aspect
        for ri in range(2):
            if time.time() - start_time > time_budget:
                break

            if ri == 0:
                cols, rows = grid_configs[ai % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, seed=ai)
            else:
                cols, rows = grid_configs[(ai + 3) % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, jitter=0.4, seed=ai + 100)

            x, y, val = _optimize_for_aspect(W, H, start_x, start_y,
                                             start_time, time_budget, rng)

            if val > best_val:
                best_val = val
                r = _compute_radii_vec(x, y, W, H)
                if _validate(x, y, r, W, H):
                    best_res = np.column_stack((x, y, r))

    if best_res is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_res
