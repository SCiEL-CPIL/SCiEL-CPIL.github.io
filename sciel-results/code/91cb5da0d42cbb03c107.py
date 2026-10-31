"""Circle packing in perimeter-4 rectangle: Multi-start SLSQP + constraint-based refinement + smooth polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
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


def _smooth_objective(z, penalty_weight=8000.0, beta=300.0):
    """Smooth penalty-based objective using soft-min for overlap with adaptive weighting."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0.001):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty with smooth soft-min
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap_raw = r_sum_pairs - d_pairs

    # Smooth ReLU: soft_relu(t) = log(1 + exp(beta*t))/beta
    # Scale penalty by overlap magnitude for stronger response to large violations
    soft_relu = np.logaddexp(0, beta * overlap_raw) / beta
    overlap_magnitude = np.maximum(overlap_raw, 0.0)
    adaptive_weight = 1.0 + 5.0 * overlap_magnitude
    obj += penalty_weight * np.sum(soft_relu * adaptive_weight)

    # Boundary penalty with smooth soft-ReLU
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    soft_left = np.logaddexp(0, beta * left_viol) / beta
    soft_right = np.logaddexp(0, beta * right_viol) / beta
    soft_bottom = np.logaddexp(0, beta * bottom_viol) / beta
    soft_top = np.logaddexp(0, beta * top_viol) / beta

    # Adaptive boundary weighting
    boundary_viol = np.maximum(
        np.maximum(np.maximum(left_viol, right_viol), bottom_viol), top_viol
    )
    boundary_weight = 1.0 + 5.0 * boundary_viol

    obj += penalty_weight * (
        np.sum(soft_left * boundary_weight) +
        np.sum(soft_right * boundary_weight) +
        np.sum(soft_bottom * boundary_weight) +
        np.sum(soft_top * boundary_weight)
    )

    # Aspect ratio constraint (smooth)
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _constraint_objective(z):
    """Objective for constraint-based SLSQP: maximize sum of radii."""
    r = z[2::3][:N]
    return -np.sum(r)


def _build_constraints():
    """Build constraint functions for SLSQP."""
    constraints = []

    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            def make_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    return constraints


def _force_relax(z, max_iter=200, time_limit=None):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003
    start = time.time()

    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break

        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
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
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-10:
            f_pairs = overlap * 10.0
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

        if max_overlap < 1e-9:
            break

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        z[0::3][:n] = x
        z[1::3][:n] = y

    return z


def _expand_radii(x, y, r, w, h, time_limit=None):
    """Iteratively expand radii to maximum valid values using vectorized ops."""
    n = N
    start = time.time()
    improved = True
    max_passes = 20

    for _pass in range(max_passes):
        if time_limit is not None and time.time() > time_limit:
            break
        if not improved:
            break
        improved = False

        for i in range(n):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall

            # Max radius from other circles (vectorized)
            dx = x[i] - x
            dy = y[i] - y
            d = np.sqrt(dx * dx + dy * dy)
            d[i] = np.inf  # ignore self
            r_allowed = d - r
            r_allowed[i] = np.inf
            r_max = min(r_max, np.min(r_allowed))

            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-8:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(12):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

    return r


def _polish_positions(x, y, r, w, h, time_limit=None):
    """Polish positions using force-directed method with adaptive step size."""
    n = N
    start = time.time()
    dt = 0.001
    max_iters = 800

    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 30
        fx += right_viol * 30
        fy -= bottom_viol * 30
        fy += top_viol * 30

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
        if max_overlap > 1e-10:
            f_pairs = overlap * 12.0
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

        # Adaptive step size: reduce if forces are large
        force_mag = np.sqrt(np.sum(fx ** 2) + np.sum(fy ** 2))
        if force_mag > 0.5:
            dt_current = dt * 0.5
        elif force_mag < 0.01:
            dt_current = dt * 1.2
        else:
            dt_current = dt

        x += dt_current * fx
        y += dt_current * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Decay step size
        dt *= 0.998

        if max_overlap < 1e-9 and force_mag < 0.005:
            break

    return x, y


def _slsqp_constraint_refine(z, time_budget):
    """SLSQP with explicit constraints for high-precision refinement."""
    n = N
    constraints = _build_constraints()
    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    maxiter = int(min(600, time_budget * 100))
    if maxiter < 100:
        maxiter = 100

    try:
        res = minimize(_constraint_objective, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 120})
        return res.x
    except Exception:
        return None


def _slsqp_smooth(z, time_budget):
    """SLSQP with smooth penalty-based objective."""
    maxiter = int(min(500, time_budget * 80))
    if maxiter < 150:
        maxiter = 150

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    try:
        res = minimize(_smooth_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 100})
        return res.x
    except Exception:
        return None


def _generate_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    side = int(np.ceil(np.sqrt(N)))

    # Grid-based initial guesses with different aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8, 1.4, 0.7]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.4
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    # Random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        w0 = rng.uniform(0.55, 1.45)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.04, 0.11)
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

    def check_and_update(z):
        nonlocal best_value, best_result
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            # Try slight shrink
            for factor in [0.99, 0.98, 0.95]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    val = float(np.sum(r_trial))
                    if val > best_value:
                        best_value = val
                        z_new = z.copy()
                        z_new[2::3][:N] = r_trial
                        best_result = z_new
                    return
            return

        val = float(np.sum(r))
        if val > best_value:
            best_value = val
            best_result = z.copy()

    guesses = _generate_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break

        # Force relax to resolve initial overlaps
        z_relaxed = _force_relax(z0, max_iter=200, time_limit=deadline - 35)

        # Smooth penalty-based SLSQP
        remaining = deadline - (time.time() - start_time) - 25
        if remaining > 5:
            z_opt = _slsqp_smooth(z_relaxed, remaining)
            if z_opt is not None:
                check_and_update(z_opt)

        # If we have a valid result, do constraint-based refinement
        if best_result is not None and time.time() < deadline - 30:
            remaining = deadline - (time.time() - start_time) - 18
            if remaining > 5:
                z_refined = _slsqp_constraint_refine(z_opt if z_opt is not None else z_relaxed, remaining)
                if z_refined is not None:
                    check_and_update(z_refined)

    # Phase 2: Polish the best result with multiple passes
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Multiple polish-expand cycles
        for _cycle in range(3):
            if time.time() > deadline - 8:
                break

            # Position polish
            x, y = _polish_positions(x, y, r, w, h, time_limit=deadline - 8)

            # Expand radii
            r = _expand_radii(x, y, r, w, h, time_limit=deadline - 8)

            # Reconstruct and check
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:N] = x
            z_new[1::3][:N] = y
            z_new[2::3][:N] = r
            z_new[-1] = w
            check_and_update(z_new)

            # Update local variables from best if it improved
            if best_result is not None:
                x = best_result[0::3][:N].copy()
                y = best_result[1::3][:N].copy()
                r = best_result[2::3][:N].copy()
                w = best_result[-1]
                h = 2.0 - w

        # Final constraint refinement if time permits
        if time.time() < deadline - 5:
            remaining = deadline - (time.time() - start_time) - 2
            if remaining > 3:
                z_final = _slsqp_constraint_refine(z_new, remaining)
                if z_final is not None:
                    check_and_update(z_final)

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
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
