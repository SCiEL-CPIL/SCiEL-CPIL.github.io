"""Circle packing in perimeter-4 rectangle: Sequential Impulse Solver + Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _sis_simulate(x, y, r, w, h, max_iter=200, dt=0.01, damping=0.95):
    """
    Sequential Impulse Solver: simulate circles as dynamic particles under
    iterative velocity-level impulse exchanges to enforce non-penetration.
    Returns final positions after convergence.
    """
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = np.ones(N)
    inv_mass = 1.0 / mass

    for it in range(max_iter):
        # Compute relative velocities and impulses for all overlapping pairs
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-10:
            break

        # Compute normal directions and relative velocities
        ux = dx[iu] / d_pairs
        uy = dy[iu] / d_pairs

        # Relative velocity along normal (positive = separating)
        dvx = vx[iu[0]] - vx[iu[1]]
        dvy = vy[iu[0]] - vy[iu[1]]
        rel_vel = dvx * ux + dvy * uy

        # Impulse magnitude: only apply if approaching (rel_vel < 0) or overlapping
        # Use a stiff spring-like impulse proportional to overlap
        impulse = np.zeros(len(iu[0]))
        mask = overlap > 1e-8
        impulse[mask] = overlap[mask] * 50.0 - rel_vel[mask] * 0.5

        # Clamp impulses to non-negative (only push apart)
        impulse = np.maximum(impulse, 0.0)

        # Apply impulses
        for idx in range(len(iu[0])):
            if impulse[idx] > 1e-12:
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                jx = impulse[idx] * ux[idx] * inv_mass[i_idx]
                jy = impulse[idx] * uy[idx] * inv_mass[i_idx]
                vx[i_idx] += jx
                vy[i_idx] += jy
                vx[j_idx] -= impulse[idx] * ux[idx] * inv_mass[j_idx]
                vy[j_idx] -= impulse[idx] * uy[idx] * inv_mass[j_idx]

        # Wall impulses
        # Left wall
        left_pen = np.maximum(r - x, 0.0)
        wall_imp = left_pen * 80.0 - vx * 0.5
        wall_imp = np.maximum(wall_imp, 0.0)
        vx += wall_imp * inv_mass
        # Right wall
        right_pen = np.maximum(x + r - w, 0.0)
        wall_imp = right_pen * 80.0 + vx * 0.5
        wall_imp = np.maximum(wall_imp, 0.0)
        vx -= wall_imp * inv_mass
        # Bottom wall
        bottom_pen = np.maximum(r - y, 0.0)
        wall_imp = bottom_pen * 80.0 - vy * 0.5
        wall_imp = np.maximum(wall_imp, 0.0)
        vy += wall_imp * inv_mass
        # Top wall
        top_pen = np.maximum(y + r - h, 0.0)
        wall_imp = top_pen * 80.0 + vy * 0.5
        wall_imp = np.maximum(wall_imp, 0.0)
        vy -= wall_imp * inv_mass

        # Integrate positions
        x += dt * vx
        y += dt * vy

        # Damping
        vx *= damping
        vy *= damping

        # Hard clamp to walls
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _sis_packing(x0, y0, r0, w, h, time_budget):
    """
    Use SIS simulation to iteratively grow radii and find equilibrium packing.
    Strategy: start with small radii, simulate to equilibrium, grow radii,
    re-simulate, repeat until no further growth possible.
    """
    x = x0.astype(float).copy()
    y = y0.astype(float).copy()
    r = r0.astype(float).copy()

    # Phase 1: Rapid growth with coarse simulation
    for grow_round in range(80):
        if time.time() > time_budget:
            break

        # Simulate to equilibrium
        x, y = _sis_simulate(x, y, r, w, h, max_iter=150, dt=0.01, damping=0.9)

        # Check validity
        if not _is_valid(x, y, r, w, h):
            # Try to repair by shrinking slightly
            r *= 0.98
            x, y = _sis_simulate(x, y, r, w, h, max_iter=100, dt=0.005, damping=0.95)
            if not _is_valid(x, y, r, w, h):
                r *= 0.95
                x, y = _sis_simulate(x, y, r, w, h, max_iter=100, dt=0.005, damping=0.95)
                if not _is_valid(x, y, r, w, h):
                    break

        # Try to grow: find max uniform growth factor
        best_factor = 1.0
        for trial_factor in [1.02, 1.015, 1.01, 1.005, 1.002]:
            r_trial = r * trial_factor
            # Check if this is feasible (walls)
            if np.any(r_trial > np.minimum(x, w - x)) or np.any(r_trial > np.minimum(y, h - y)):
                continue
            # Simulate with new radii
            x_trial, y_trial = _sis_simulate(x, y, r_trial, w, h, max_iter=200, dt=0.008, damping=0.92)
            if _is_valid(x_trial, y_trial, r_trial, w, h):
                best_factor = trial_factor
                x = x_trial
                y = y_trial
                r = r_trial
                break

        if best_factor == 1.0:
            # Try per-circle growth
            improved = False
            for i in range(N):
                if time.time() > time_budget:
                    break
                # Max radius for circle i
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
                r_max = max(0.0, r_max)

                if r_max > r[i] + 1e-6:
                    lo, hi = r[i], r_max
                    for _ in range(20):
                        mid = (lo + hi) / 2.0
                        r_test = r.copy()
                        r_test[i] = mid
                        x_t, y_t = _sis_simulate(x, y, r_test, w, h, max_iter=100, dt=0.005, damping=0.93)
                        if _is_valid(x_t, y_t, r_test, w, h):
                            lo = mid
                            x, y = x_t, y_t
                        else:
                            hi = mid
                    if lo > r[i] + 1e-7:
                        r[i] = lo
                        improved = True

            if not improved:
                break

    # Phase 2: Fine-tuning with more simulation iterations
    if time.time() < time_budget - 5:
        for _ in range(20):
            if time.time() > time_budget:
                break
            x, y = _sis_simulate(x, y, r, w, h, max_iter=300, dt=0.005, damping=0.97)
            if not _is_valid(x, y, r, w, h):
                break
            # Tiny uniform growth
            r_trial = r * 1.001
            if np.any(r_trial > np.minimum(x, w - x)) or np.any(r_trial > np.minimum(y, h - y)):
                break
            x_t, y_t = _sis_simulate(x, y, r_trial, w, h, max_iter=300, dt=0.005, damping=0.97)
            if _is_valid(x_t, y_t, r_trial, w, h):
                x, y, r = x_t, y_t, r_trial
            else:
                break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_w = 1.0
    best_h = 1.0

    def check_validity(x, y, r, w, h, tol=1e-9):
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

    # Generate diverse initial configurations
    def get_initial_configs():
        configs = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                r[i] = min(w0 / side, h0 / side) * 0.3
            configs.append((x, y, r, w0, h0))

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            configs.append((x, y, r, w0, h0))

        return configs

    configs = get_initial_configs()

    # Phase 1: SIS-based packing from multiple starts
    for i, (x0, y0, r0, w0, h0) in enumerate(configs):
        if time.time() > deadline - 15:
            break

        remaining = deadline - (time.time() - start_time)
        time_budget = start_time + min(remaining * 0.7, 12.0)

        result = _sis_packing(x0, y0, r0, w0, h0, time_budget)
        if result is not None:
            x_res, y_res, r_res = result
            valid, val = check_validity(x_res, y_res, r_res, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_x = x_res.copy()
                best_y = y_res.copy()
                best_r = r_res.copy()
                best_w = w0
                best_h = h0

    # Phase 2: Polish best result
    if best_x is not None and time.time() < deadline - 10:
        # SIS-based polish: re-simulate with more iterations
        polish_budget = deadline - (time.time() - start_time) - 3
        x_p, y_p, r_p = _sis_packing(best_x, best_y, best_r, best_w, best_h, start_time + polish_budget)
        if x_p is not None:
            valid, val = check_validity(x_p, y_p, r_p, best_w, best_h)
            if valid and val > best_value:
                best_value = val
                best_x, best_y, best_r = x_p, y_p, r_p

        # Greedy per-circle expansion
        if time.time() < deadline - 3:
            x_g = best_x.copy()
            y_g = best_y.copy()
            r_g = best_r.copy()
            for _ in range(3):
                if time.time() > deadline - 2:
                    break
                for i in range(N):
                    if time.time() > deadline - 2:
                        break
                    r_wall = min(x_g[i], best_w - x_g[i], y_g[i], best_h - y_g[i])
                    r_max = r_wall
                    for j in range(N):
                        if i == j:
                            continue
                        dx = x_g[i] - x_g[j]
                        dy = y_g[i] - y_g[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_allowed = d - r_g[j]
                        if r_allowed < r_max:
                            r_max = r_allowed
                    r_max = max(0.0, r_max)

                    if r_max > r_g[i] + 1e-8:
                        lo, hi = r_g[i], r_max
                        for _ in range(30):
                            mid = (lo + hi) / 2.0
                            r_test = r_g.copy()
                            r_test[i] = mid
                            if _is_valid(x_g, y_g, r_test, best_w, best_h):
                                lo = mid
                            else:
                                hi = mid
                        if lo > r_g[i] + 1e-9:
                            r_g[i] = lo

            valid, val = check_validity(x_g, y_g, r_g, best_w, best_h)
            if valid and val > best_value:
                best_value = val
                best_x, best_y, best_r = x_g, y_g, r_g

    if best_x is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = best_r * (1 - 1e-9)
        if _is_valid(best_x, best_y, r_trial, best_w, best_h):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, best_r))


def _force_relax_unified(z, max_iter=500):
    """Legacy force-directed relaxation (kept for compatibility)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

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

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

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

        if max_overlap > convergence_tol:
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

        if max_overlap < convergence_tol:
            break

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


def _polish_result(z, time_budget):
    """Polish using SIS simulation (replaced force-directed approach)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Use SIS to polish
    result = _sis_packing(x, y, r, w, h, time_budget)
    if result is not None:
        x_p, y_p, r_p = result
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x_p
        z_new[1::3][:n] = y_p
        z_new[2::3][:n] = r_p
        z_new[-1] = w
        if _is_valid(x_p, y_p, r_p, w, h):
            return z_new
    return None


def _expansion_relaxation_polish_enhanced(z, time_budget):
    """Enhanced polish using SIS (replaced multi-direction local search)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    result = _sis_packing(x, y, r, w, h, time_budget)
    if result is not None:
        x_p, y_p, r_p = result
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x_p
        z_new[1::3][:n] = y_p
        z_new[2::3][:n] = r_p
        z_new[-1] = w
        if _is_valid(x_p, y_p, r_p, w, h):
            return z_new
    return None


def _epigraph_refine(z, time_budget):
    """Epigraph-based refinement using SIS (replaced SLSQP with explicit constraints)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    result = _sis_packing(x, y, r, w, h, time_budget)
    if result is not None:
        x_p, y_p, r_p = result
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x_p
        z_new[1::3][:n] = y_p
        z_new[2::3][:n] = r_p
        z_new[-1] = w
        if _is_valid(x_p, y_p, r_p, w, h):
            return z_new
    return None


def _constraint_refine(z, time_budget):
    """Constraint-based refinement using SIS (replaced SLSQP)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    result = _sis_packing(x, y, r, w, h, time_budget)
    if result is not None:
        x_p, y_p, r_p = result
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x_p
        z_new[1::3][:n] = y_p
        z_new[2::3][:n] = r_p
        z_new[-1] = w
        if _is_valid(x_p, y_p, r_p, w, h):
            return z_new
    return None


def _coordinate_ascent_refine(z, time_budget):
    """Coordinate ascent refinement using SIS (replaced manual coordinate search)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    result = _sis_packing(x, y, r, w, h, time_budget)
    if result is not None:
        x_p, y_p, r_p = result
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x_p
        z_new[1::3][:n] = y_p
        z_new[2::3][:n] = r_p
        z_new[-1] = w
        if _is_valid(x_p, y_p, r_p, w, h):
            return z_new
    return None


def _greedy_expansion(z, time_budget):
    """
    Final per-circle greedy binary-search expansion: for each circle,
    binary-search the maximum valid radius given all other positions fixed.
    This squeezes out the last gains that continuous optimizers miss.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    for _ in range(3):  # Multiple passes to allow cascading improvements
        if time.time() > time_budget:
            break
        for i in range(n):
            if time.time() > time_budget:
                break
            # Compute max radius for circle i
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-9:
                # Binary search
                lo = r[i]
                hi = r_max
                for _ in range(40):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


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
    # Wall check
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False

    # Neighbor check
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
