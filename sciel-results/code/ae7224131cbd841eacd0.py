"""Circle packing in perimeter-4 rectangle: improved SLSQP with contact-graph polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # --- Objective function for SLSQP ---
    def objective(z, penalty_weight=500.0):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Vectorized Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_req = r[:, None] + r[None, :]

        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        violations = np.maximum(0, dist_req - dist)
        violations = violations * mask
        obj += penalty_weight * np.sum(violations ** 2)

        # Vectorized Boundary penalty
        viol_lx = np.maximum(0, r - x)
        viol_by = np.maximum(0, r - y)
        viol_rx = np.maximum(0, x - (w - r))
        viol_ty = np.maximum(0, y - (h - r))

        obj += penalty_weight * (np.sum(viol_lx ** 2) + np.sum(viol_by ** 2) +
                                 np.sum(viol_rx ** 2) + np.sum(viol_ty ** 2))

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0

        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0

        # Overlap checks (Vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        violations = (d2 < r_sum ** 2 - tol) & mask
        if np.any(violations):
            return False, 0.0

        val = float(np.sum(r))
        return True, val

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []

        # Grid-based starts with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
        side = int(np.ceil(np.sqrt(N)))

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = []
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(12 - len(guesses)):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = []
            for i in range(N):
                cx = rng.uniform(0.1, w0 - 0.1)
                cy = rng.uniform(0.1, h0 - 0.1)
                r0 = rng.uniform(0.05, 0.12)
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 5:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(150, remaining * 35)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 100})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Contact-graph-aware polish ---
    if best_result is not None and time.time() < deadline - 20:
        polished = _contact_graph_polish(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # --- Phase 3: High-precision SLSQP refinement ---
    if best_result is not None and time.time() < deadline - 5:
        refined = _high_precision_refine(best_result, deadline - (time.time() - start_time))
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    # --- Phase 4: Final radius growth ---
    if best_result is not None and time.time() < deadline - 2:
        final = _final_radius_growth(best_result, deadline - (time.time() - start_time))
        if final is not None:
            valid, val = check_validity(final)
            if valid and val > best_value:
                best_value = val
                best_result = final

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

    # Final safety: ensure strict validity with small shrink
    for _ in range(50):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0005

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Vectorized Wall repulsion
        fx += np.where(x - r < 0, (x - r) * 25, 0)
        fy += np.where(y - r < 0, (y - r) * 25, 0)
        fx -= np.where(x + r > w, (x + r - w) * 25, 0)
        fy -= np.where(y + r > h, (y + r - h) * 25, 0)

        # Vectorized Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        min_d = r[:, None] + r[None, :]

        dist_safe = np.where(dist > 1e-12, dist, 1e-12)
        overlap = np.maximum(0, min_d - dist)
        force_mag = (overlap * 12) / dist_safe

        fx_diff = dx * force_mag
        fy_diff = dy * force_mag

        fx += np.sum(fx_diff, axis=1)
        fy += np.sum(fy_diff, axis=1)

        # Adaptive step size
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        if max_force > 0.5:
            dt = max(min_dt, 0.002)
        elif max_force > 0.1:
            dt = max(min_dt, 0.004)
        else:
            dt = 0.005

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z


def _contact_graph_polish(z, time_budget):
    """Polish using contact-graph-aware force-directed relaxation with radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0005
    max_iter = int(min(2000, time_budget * 60))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (stronger for precision)
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 40
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 40
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 40
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 40

        # Pairwise repulsion (stronger for precision)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20

        # Adaptive step
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        if max_force > 0.1:
            dt = 0.0003
        else:
            dt = 0.0005

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0002
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _high_precision_refine(z, time_budget):
    """High-precision SLSQP refinement on a valid packing."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        pw = 2000.0  # Tighter penalties

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_req = r[:, None] + r[None, :]

        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        violations = np.maximum(0, dist_req - dist)
        violations = violations * mask
        obj += pw * np.sum(violations ** 2)

        viol_lx = np.maximum(0, r - x)
        viol_by = np.maximum(0, r - y)
        viol_rx = np.maximum(0, x - (w - r))
        viol_ty = np.maximum(0, y - (h - r))

        obj += pw * (np.sum(viol_lx ** 2) + np.sum(viol_by ** 2) +
                     np.sum(viol_rx ** 2) + np.sum(viol_ty ** 2))

        return obj

    maxiter = int(min(600, time_budget * 80))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=[(0, 2)] * (3 * n) + [(0.3, 1.7)],
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 120})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _final_radius_growth(z, time_budget):
    """Final aggressive radius growth with validity checking."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Try uniform growth first
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Try per-circle growth
    for i in range(n):
        if time.time() > time_budget:
            break
        for _ in range(100):
            r_trial = r.copy()
            r_trial[i] *= 1.0001
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
            else:
                break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False

    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    violations = (d2 < r_sum ** 2 - tol) & mask
    if np.any(violations):
        return False

    return True
