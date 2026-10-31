"""Circle packing in perimeter-4 rectangle: Multi-start L-BFGS-B + vectorized expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
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


def _vectorized_objective(z, w, h, penalty=100.0):
    """
    Vectorized objective: minimize -sum(r) + penalty * constraint violations.
    z layout: [x_0..x_20, y_0..y_20, r_0..r_20] (63 elements)
    w, h are fixed per call.
    """
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]

    # Base objective: maximize sum of radii
    val = -np.sum(r)

    # Boundary violations (squared)
    viol = 0.0
    # Left wall
    d = r - x
    viol += np.sum(np.maximum(d, 0.0) ** 2)
    # Right wall
    d = x + r - w
    viol += np.sum(np.maximum(d, 0.0) ** 2)
    # Bottom wall
    d = r - y
    viol += np.sum(np.maximum(d, 0.0) ** 2)
    # Top wall
    d = y + r - h
    viol += np.sum(np.maximum(d, 0.0) ** 2)

    # Pairwise overlap violations (vectorized via broadcasting)
    # dx[i,j] = x[i] - x[j], etc.
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dmat = np.sqrt(dx * dx + dy * dy + 1e-12)
    rsum = r[:, None] + r[None, :]
    overlap = np.maximum(rsum - dmat, 0.0)
    # Only count upper triangle (avoid double counting)
    iu = np.triu_indices(N, k=1)
    viol += np.sum(overlap[iu] ** 2)

    val += penalty * viol
    return val


def _lbfgsb_optimize(x0, y0, r0, w, h, time_budget, penalty_start=50.0):
    """Run L-BFGS-B optimization with adaptive penalty."""
    n = N
    z0 = np.concatenate([x0, y0, r0])

    bounds = []
    for i in range(n):
        bounds.append((0.0, w))
    for i in range(n):
        bounds.append((0.0, h))
    for i in range(n):
        bounds.append((1e-4, min(w, h) / 2))

    penalty = penalty_start
    best_z = z0.copy()
    best_val = _vectorized_objective(z0, w, h, penalty)

    elapsed = time.time()
    while time.time() < time_budget:
        try:
            res = minimize(
                lambda z: _vectorized_objective(z, w, h, penalty),
                z0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            z0 = res.x
            cur_val = _vectorized_objective(z0, w, h, penalty)
            if cur_val < best_val:
                best_val = cur_val
                best_z = z0.copy()
            # Reduce penalty if violations are small
            x = z0[0:N]
            y = z0[N:2*N]
            r = z0[2*N:3*N]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dmat = np.sqrt(dx * dx + dy * dy + 1e-12)
            rsum = r[:, None] + r[None, :]
            overlap = np.maximum(rsum - dmat, 0.0)
            iu = np.triu_indices(N, k=1)
            max_ov = np.max(overlap[iu]) if N > 1 else 0.0
            if max_ov < 1e-6:
                break
            penalty *= 0.5
            if penalty < 1.0:
                break
        except Exception:
            break

    x = best_z[0:N]
    y = best_z[N:2*N]
    r = best_z[2*N:3*N]
    return x, y, r


def _expand_radii_vectorized(x, y, r, w, h, max_rounds=50, time_budget=None):
    """Vectorized radius expansion: iteratively grow each circle to max allowed."""
    for rnd in range(max_rounds):
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            # Max from walls
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            # Max from other circles
            dx = x[i] - x
            dy = y[i] - y
            d = np.sqrt(dx * dx + dy * dy)
            # Exclude self
            d[i] = np.inf
            r_allowed = d - r
            r_allowed[i] = np.inf
            r_max = min(r_max, np.min(r_allowed))
            r_max = max(r_max, 0.0)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return r


def _polish_positions(x, y, r, w, h, time_budget):
    """
    Local search: try small random perturbations of positions, then re-expand radii.
    Keep if sum(r) improves.
    """
    x_best = x.copy()
    y_best = y.copy()
    r_best = r.copy()
    best_sum = np.sum(r_best)

    rng = np.random.default_rng(42)

    for iteration in range(200):
        if time.time() > time_budget:
            break

        # Pick a random subset of circles to perturb
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)

        x_try = x_best.copy()
        y_try = y_best.copy()

        for i in idx:
            angle = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(0.002, 0.02)
            x_try[i] += mag * np.cos(angle)
            y_try[i] += mag * np.sin(angle)
            # Clamp to valid range
            x_try[i] = np.clip(x_try[i], r_best[i], w - r_best[i])
            y_try[i] = np.clip(y_try[i], r_best[i], h - r_best[i])

        # Re-expand radii
        r_try = _expand_radii_vectorized(x_try, y_try, r_best.copy(), w, h,
                                         max_rounds=10, time_budget=time_budget)

        new_sum = np.sum(r_try)
        if new_sum > best_sum + 1e-10:
            x_best = x_try
            y_best = y_try
            r_best = r_try
            best_sum = new_sum

    return x_best, y_best, r_best


def _grid_init(w, h, seed=0):
    """Generate a grid-based initial placement."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w + rng.uniform(-0.02, 0.02)
        y[i] = (row + 0.5) / side * h + rng.uniform(-0.02, 0.02)
        r[i] = min(w, h) / side * 0.35
    # Clamp
    for i in range(N):
        r[i] = min(r[i], min(x[i], w - x[i], y[i], h - y[i]) * 0.9)
        x[i] = np.clip(x[i], r[i], w - r[i])
        y[i] = np.clip(y[i], r[i], h - r[i])
    return x, y, r


def _random_init(w, h, seed=0):
    """Generate a random initial placement."""
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        ri = rng.uniform(0.02, min(w, h) / 4)
        xi = rng.uniform(ri, w - ri)
        yi = rng.uniform(ri, h - ri)
        # Check against already placed
        valid = True
        for j in range(i):
            dx = xi - x[j]
            dy = yi - y[j]
            if dx * dx + dy * dy < (ri + r[j]) ** 2:
                valid = False
                break
        if valid:
            x[i] = xi
            y[i] = yi
            r[i] = ri
        else:
            # Place with tiny radius
            ri = 0.015
            xi = rng.uniform(ri, w - ri)
            yi = rng.uniform(ri, h - ri)
            x[i] = xi
            y[i] = yi
            r[i] = ri
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = -1.0
    best_result = None

    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Aspect ratios to explore
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67,
                     1.6, 0.6, 1.75, 0.55, 2.0, 0.5, 1.8, 0.55, 1.4, 0.85,
                     1.15, 0.95, 1.05, 1.2, 0.83]

    # Phase 1: Multi-start L-BFGS-B across aspect ratios
    for wi, w in enumerate(aspect_ratios):
        if time.time() > deadline - 40:
            break
        h = 2.0 - w
        if h < 0.25:
            continue

        # Multiple initializations per aspect ratio
        n_starts = 3 if w > 0.5 and w < 1.5 else 2
        for si in range(n_starts):
            if time.time() > deadline - 35:
                break

            if si == 0:
                x0, y0, r0 = _grid_init(w, h, seed=wi * 10 + si)
            elif si == 1:
                x0, y0, r0 = _random_init(w, h, seed=wi * 10 + si + 100)
            else:
                x0, y0, r0 = _grid_init(w, h, seed=wi * 10 + si + 200)

            # L-BFGS-B optimization
            x1, y1, r1 = _lbfgsb_optimize(x0, y0, r0, w, h,
                                           time_budget=deadline - 30,
                                           penalty_start=50.0)

            # Vectorized radius expansion
            r1 = _expand_radii_vectorized(x1, y1, r1, w, h,
                                          max_rounds=30, time_budget=deadline - 10)

            update_best(x1, y1, r1, w, h)

            # Position polish (local search)
            if time.time() < deadline - 20:
                x2, y2, r2 = _polish_positions(x1, y1, r1, w, h,
                                                time_budget=deadline - 10)
                r2 = _expand_radii_vectorized(x2, y2, r2, w, h,
                                              max_rounds=20, time_budget=deadline - 5)
                update_best(x2, y2, r2, w, h)

    # Phase 2: Additional random starts with different seeds
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        w = 0.7 + 0.6 * (seed / 9.0)  # Spread across 0.7 to 1.3
        h = 2.0 - w
        if h < 0.3:
            continue

        x0, y0, r0 = _random_init(w, h, seed=seed + 500)
        x1, y1, r1 = _lbfgsb_optimize(x0, y0, r0, w, h,
                                       time_budget=deadline - 15,
                                       penalty_start=80.0)
        r1 = _expand_radii_vectorized(x1, y1, r1, w, h,
                                      max_rounds=20, time_budget=deadline - 5)
        update_best(x1, y1, r1, w, h)

        if time.time() < deadline - 10:
            x2, y2, r2 = _polish_positions(x1, y1, r1, w, h,
                                            time_budget=deadline - 5)
            r2 = _expand_radii_vectorized(x2, y2, r2, w, h,
                                          max_rounds=15, time_budget=deadline - 2)
            update_best(x2, y2, r2, w, h)

    # Phase 3: Deep polish on the best result
    if best_result is not None and time.time() < deadline - 15:
        x, y, r, w, h = best_result

        # Iterative polish
        for it in range(5):
            if time.time() > deadline - 5:
                break
            x, y, r = _polish_positions(x, y, r, w, h, time_budget=deadline - 3)
            r = _expand_radii_vectorized(x, y, r, w, h,
                                         max_rounds=20, time_budget=deadline - 2)
            update_best(x, y, r, w, h)

        # Final L-BFGS-B refinement on best
        if time.time() < deadline - 8:
            x_f, y_f, r_f = _lbfgsb_optimize(x, y, r, w, h,
                                              time_budget=deadline - 3,
                                              penalty_start=200.0)
            r_f = _expand_radii_vectorized(x_f, y_f, r_f, w, h,
                                           max_rounds=30, time_budget=deadline - 1)
            update_best(x_f, y_f, r_f, w, h)

    # Fallback
    if best_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w
            y[i] = (row + 0.5) / side * h
            r[i] = min(w, h) / side * 0.35
        best_result = (x, y, r, w, h)

    x, y, r, w, h = best_result

    # Final safety: shrink slightly if needed
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
