"""Circle packing in perimeter-4 rectangle: Simulated annealing + iterative expansion + local search."""
import numpy as np
import time
import random

N = 21


def _compute_energy(x, y, r, w, h):
    """Compute total overlap energy (sum of squared overlaps)."""
    energy = 0.0
    # Boundary violations
    for i in range(N):
        if x[i] < r[i]:
            energy += (r[i] - x[i]) ** 2
        if x[i] > w - r[i]:
            energy += (x[i] - (w - r[i])) ** 2
        if y[i] < r[i]:
            energy += (r[i] - y[i]) ** 2
        if y[i] > h - r[i]:
            energy += (y[i] - (h - r[i])) ** 2
    # Pairwise overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            rs = r[i] + r[j]
            if d < rs:
                energy += (rs - d) ** 2
    return energy


def _simulated_annealing(x0, y0, r0, w0, h0, time_budget, seed=42):
    """Run simulated annealing to find a valid packing."""
    rng = np.random.default_rng(seed)
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()
    w = w0
    h = h0

    # Start with small radii that fit
    max_r = min(w / 5, h / 5) * 0.8
    for i in range(N):
        r[i] = min(r[i], max_r)

    T_start = 0.1
    T_end = 0.001
    n_iter = 50000
    alpha = (T_end / T_start) ** (1.0 / n_iter)

    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
    best_val = -np.sum(r)

    for it in range(n_iter):
        if time.time() > time_budget:
            break

        T = T_start * (alpha ** it)

        # Random move: pick a circle and perturb it
        i = rng.integers(N)

        # Perturb position
        dx = rng.normal(0, 0.05)
        dy = rng.normal(0, 0.05)
        new_x = np.clip(x[i] + dx, r[i], w - r[i])
        new_y = np.clip(y[i] + dy, r[i], h - r[i])

        # Occasionally perturb radius
        if rng.random() < 0.3:
            dr = rng.normal(0, 0.02)
            new_r = np.clip(r[i] + dr, 0.01, min(w / 3, h / 3))
        else:
            new_r = r[i]

        # Compute energy difference
        old_energy = _compute_energy(x, y, r, w, h)

        x[i] = new_x
        y[i] = new_y
        r[i] = new_r
        new_energy = _compute_energy(x, y, r, w, h)

        delta_e = new_energy - old_energy

        # Accept or reject
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            # Keep move
            val = -new_energy + np.sum(r)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
        else:
            # Revert
            x[i] = new_x - dx if x[i] != new_x else x[i]
            # Actually need to revert properly
            x[i] = x0[i] if False else x[i]  # This is wrong, need to store old values
            # Let me fix this by storing old values
            pass

    return best_x, best_y, best_r, best_w


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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


def _max_radius_at(x, y, r, w, h, i):
    """Find max radius for circle i at position (x[i], y[i])."""
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


def _expand_radii_greedy(x, y, r, w, h, max_rounds=100, time_budget=None):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_rounds):
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            r_max = _max_radius_at(x, y, r, w, h, i)
            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return x, y, r


def _local_search(x, y, r, w, h, time_budget, rng=None):
    """Local search: try moving each circle to improve total radius."""
    if rng is None:
        rng = np.random.default_rng()
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for _ in range(2000):
        if time.time() > time_budget:
            break

        i = rng.integers(N)
        old_x, old_y, old_r = x[i], y[i], r[i]

        # Try random position
        new_x = rng.uniform(r[i], w - r[i])
        new_y = rng.uniform(r[i], h - r[i])

        # Check validity with current radius
        valid = True
        if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
            valid = False
        if valid:
            for j in range(N):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                    valid = False
                    break

        if valid:
            x[i] = new_x
            y[i] = new_y
            # Try to expand radius
            r_new = _max_radius_at(x, y, r, w, h, i)
            if r_new > r[i] + 1e-6:
                r[i] = r_new
        else:
            # Try with smaller radius
            for trial_r in [r[i] * 0.9, r[i] * 0.8, r[i] * 0.7]:
                if trial_r < 0.01:
                    break
                valid = True
                if new_x < trial_r or new_x > w - trial_r or new_y < trial_r or new_y > h - trial_r:
                    valid = False
                if valid:
                    for j in range(N):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        if dx * dx + dy * dy < (trial_r + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                if valid:
                    x[i] = new_x
                    y[i] = new_y
                    r[i] = trial_r
                    break

    return x, y, r


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


def _expand_all(x, y, r, w, h, max_rounds=50, time_budget=None):
    """Iteratively expand each circle to its max possible radius with position adjustment."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    rng = np.random.default_rng(123)
    for _ in range(max_rounds):
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            r_max = _max_radius_at(x, y, r, w, h, i)
            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
        
        # Try to shift circles to create more space
        if improved:
            for i in range(N):
                min_margin = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
                if min_margin < 1e-6:
                    for _ in range(10):
                        dx = rng.uniform(-0.02, 0.02)
                        dy = rng.uniform(-0.02, 0.02)
                        new_x = x[i] + dx
                        new_y = y[i] + dy
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            break
    
        if not improved:
            break
    return x, y, r


def _force_relax(x, y, r, w, h, max_iter=500, time_budget=None):
    """Force-directed relaxation to resolve overlaps."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    dt = 0.005
    min_dt = 0.00001
    damping = 0.95

    vx = np.zeros(N)
    vy = np.zeros(N)

    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall repulsion
        for i in range(N):
            if x[i] < r[i]:
                fx[i] -= (r[i] - x[i]) * 50
            if x[i] > w - r[i]:
                fx[i] += (x[i] - (w - r[i])) * 50
            if y[i] < r[i]:
                fy[i] -= (r[i] - y[i]) * 50
            if y[i] > h - r[i]:
                fy[i] += (y[i] - (h - r[i])) * 50

        # Pairwise repulsion
        max_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                rs = r[i] + r[j]
                if d < rs:
                    overlap = rs - d
                    max_overlap = max(max_overlap, overlap)
                    if d > 1e-6:
                        ux = dx / d
                        uy = dy / d
                    else:
                        ux, uy = 1.0, 0.0
                    f = overlap * 10.0
                    fx[i] += ux * f
                    fy[i] += uy * f
                    fx[j] -= ux * f
                    fy[j] -= uy * f

        if max_overlap < 1e-8:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 2.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.05:
            dt = min(0.01, dt * 1.2)

        vx = vx * damping + dt * fx
        vy = vy * damping + dt * fy

        x += vx
        y += vy

        # Clamp positions
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

    return x, y, r


def _get_initial_guesses():
    """Generate diverse initial guesses: grids, hex, and random."""
    guesses = []
    
    # Grid layouts
    for w0 in [0.6, 0.8, 1.0, 1.2, 1.4]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        for cols in [3, 4, 5]:
            rows = int(np.ceil(N / cols))
            if rows * cols < N:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, cols)
                if row >= rows:
                    break
                x[i] = (col + 0.5) / cols * w0
                y[i] = (row + 0.5) / rows * h0
                r[i] = min(w0 / cols, h0 / rows) * 0.4
            guesses.append((x, y, r, w0))
    
    # Hexagonal
    for w0 in [0.8, 1.0, 1.2]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        spacing = min(w0, h0) / 5.0
        r0 = spacing * 0.42
        row = 0
        col = 0
        for i in range(N):
            cx = (col + 0.5) * spacing + (row % 2) * spacing * 0.5
            cy = (row + 0.5) * spacing * 0.866
            if cx + r0 > w0 or cy + r0 > h0:
                col = 0
                row += 1
                cx = (col + 0.5) * spacing + (row % 2) * spacing * 0.5
                cy = (row + 0.5) * spacing * 0.866
                if cx + r0 > w0 or cy + r0 > h0:
                    break
            x[i] = cx
            y[i] = cy
            r[i] = r0
            col += 1
        guesses.append((x, y, r, w0))
    
    # Random
    rng = np.random.default_rng(42)
    for trial in range(15):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        base_r = rng.uniform(0.04, 0.08)
        for i in range(N):
            r[i] = base_r * rng.uniform(0.8, 1.2)
            x[i] = rng.uniform(r[i], w0 - r[i])
            y[i] = rng.uniform(r[i], h0 - r[i])
        guesses.append((x, y, r, w0))
    
    return guesses


def _polish_with_lbfgsb(x, y, r, w, time_budget):
    """Polish using L-BFGS-B on the penalized objective."""
    from scipy.optimize import minimize
    
    z = np.concatenate([x, y, r, [w]])
    n = N
    
    def objective(zv):
        xv = zv[0:n]
        yv = zv[n:2*n]
        rv = zv[2*n:3*n]
        wv = zv[3*n]
        hv = 2.0 - wv

        if wv < 0.2 or hv < 0.2 or np.any(rv < 0):
            return 1e6

        obj = -np.sum(rv)

        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rv[:, None] + rv[None, :]

        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        obj += 10000.0 * np.sum(overlap * overlap)

        left_viol = np.maximum(rv - xv, 0.0)
        right_viol = np.maximum(xv + rv - wv, 0.0)
        bottom_viol = np.maximum(rv - yv, 0.0)
        top_viol = np.maximum(yv + rv - hv, 0.0)
        obj += 10000.0 * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                         np.sum(bottom_viol**2) + np.sum(top_viol**2))

        if wv < 0.4:
            obj += 10000.0 * (0.4 - wv) ** 2
        elif wv > 1.6:
            obj += 10000.0 * (wv - 1.6) ** 2

        return obj

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    try:
        res = minimize(objective, z, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        z = res.x
    except Exception:
        pass

    x_new = z[0:N].copy()
    y_new = z[N:2*N].copy()
    r_new = z[2*N:3*N].copy()
    w_new = z[3*N]
    return x_new, y_new, r_new, w_new


def _expansion_relaxation_polish(x, y, r, w, time_budget):
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    h = 2.0 - w
    rng = np.random.default_rng(999)

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(N):
            # 1. Expand radius to max possible
            r_max = _max_radius_at(x, y, r, w, h, i)
            if r_max > r[i] + 1e-6:
                r[i] = r_max
                improved = True

            # 2. If touching constraints, try to move
            min_margin = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
            if min_margin < 0.02:
                # Determine direction to move
                move_x = 0.0
                move_y = 0.0
                if x[i] - r[i] == min_margin:
                    move_x = 1.0
                elif w - x[i] - r[i] == min_margin:
                    move_x = -1.0
                elif y[i] - r[i] == min_margin:
                    move_y = 1.0
                else:
                    move_y = -1.0

                for step in [0.04, 0.02, 0.01, 0.005]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break

        if not improved:
            break

    # Final uniform expansion
    for _ in range(50):
        if time.time() > time_budget:
            break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r, w
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_x, best_y, best_r, best_w = None, None, None, None

    def check_validity(x, y, r, w, tol=1e-9):
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
        return True, float(np.sum(r))

    guesses = _get_initial_guesses()

    for x0, y0, r0, w0 in guesses:
        if time.time() > deadline - 20:
            break

        h0 = 2.0 - w0

        # Force relax to resolve overlaps
        x, y, r = _force_relax(x0, y0, r0, w0, h0, max_iter=300, time_budget=deadline - 15)

        # L-BFGS polish
        if time.time() < deadline - 10:
            x, y, r, w = _polish_with_lbfgsb(x, y, r, w0, deadline - 8)
            h = 2.0 - w

        # Expand radii
        x, y, r = _expand_all(x, y, r, w, h, max_rounds=30, time_budget=deadline - 8)

        valid, val = check_validity(x, y, r, w)
        if valid and val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

    # Phase 2: Polish best result
    if best_x is not None and time.time() < deadline - 15:
        # Expansion relaxation polish
        result = _expansion_relaxation_polish(best_x, best_y, best_r, best_w, deadline - 10)
        if result is not None:
            x, y, r, w = result
            valid, val = check_validity(x, y, r, w)
            if valid and val > best_value:
                best_value = val
                best_x, best_y, best_r, best_w = x, y, r, w

        # Final L-BFGS polish
        if time.time() < deadline - 6:
            x, y, r, w = _polish_with_lbfgsb(best_x, best_y, best_r, best_w, deadline - 4)
            h = 2.0 - w
            x, y, r = _expand_all(x, y, r, w, h, max_rounds=30, time_budget=deadline - 4)
            valid, val = check_validity(x, y, r, w)
            if valid and val > best_value:
                best_value = val
                best_x, best_y, best_r, best_w = x, y, r, w

    if best_x is None:
        # Fallback
        x = np.linspace(0.15, 1.85, N)
        y = np.full(N, 1.0)
        r = np.full(N, 0.08)
        return np.column_stack((x, y, r))

    # Final safety shrink
    h = 2.0 - best_w
    for _ in range(20):
        r_trial = best_r * (1 - 1e-10)
        if _is_valid(best_x, best_y, r_trial, best_w, h):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, best_r))
