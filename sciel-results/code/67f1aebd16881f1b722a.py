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


def _soft_max(v, beta=50.0):
    """Soft maximum via log-sum-exp."""
    v_max = np.max(v)
    if v_max < -100:
        return 0.0
    return v_max + np.log(np.sum(np.exp(beta * (v - v_max)))) / beta


def _smooth_objective(z, penalty_weight=5000.0, beta=200.0):
    """Smooth penalty-based objective using soft-min for overlap."""
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
    overlap = r_sum_pairs - d_pairs
    # Use soft-max to smooth the max overlap
    obj += penalty_weight * _soft_max(overlap, beta)

    # Boundary penalty
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h
    obj += penalty_weight * (
        _soft_max(left_viol, beta) +
        _soft_max(right_viol, beta) +
        _soft_max(bottom_viol, beta) +
        _soft_max(top_viol, beta)
    )

    # Aspect ratio constraint
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


def _max_radius_for_circle(i, x, y, r, w, h):
    """Compute maximum valid radius for circle i given positions."""
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


def _expand_radii(x, y, r, w, h, time_limit=None):
    """Iteratively expand radii to maximum valid values."""
    n = N
    start = time.time()
    improved = True

    while improved:
        if time_limit is not None and time.time() > time_limit:
            break
        improved = False

        for i in range(n):
            r_max = _max_radius_for_circle(i, x, y, r, w, h)

            if r_max > r[i] + 1e-8:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(15):
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


def _expansion_relaxation_polish(x, y, r, w, h, time_limit=None):
    """Combined expansion and position optimization polish."""
    n = N
    start = time.time()
    
    # Multiple rounds of expand + local position search
    for round in range(3):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Expand radii
        r = _expand_radii(x, y, r, w, h, time_limit=time_limit)
        
        # Local position search: try small perturbations
        improved_pos = True
        while improved_pos:
            if time_limit is not None and time.time() > time_limit:
                break
            improved_pos = False
            
            for i in range(n):
                # Try 8-directional local search
                for dx_try in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                    for dy_try in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                        if dx_try == 0 and dy_try == 0:
                            continue
                        x_new = x.copy()
                        y_new = y.copy()
                        x_new[i] = x[i] + dx_try
                        y_new[i] = y[i] + dy_try
                        
                        # Check bounds
                        if x_new[i] < r[i] or x_new[i] > w - r[i]:
                            continue
                        if y_new[i] < r[i] or y_new[i] > h - r[i]:
                            continue
                        
                        # Check validity
                        if _is_valid(x_new, y_new, r, w, h):
                            # Check if this allows larger radii
                            r_test = r.copy()
                            for j in range(n):
                                r_max_j = _max_radius_for_circle(j, x_new, y_new, r_test, w, h)
                                if r_max_j > r_test[j] + 1e-7:
                                    r_test[j] = r_max_j
                            
                            if np.sum(r_test) > np.sum(r) + 1e-8:
                                x = x_new
                                y = y_new
                                r = r_test
                                improved_pos = True
                                break
                    if improved_pos:
                        break
            if improved_pos:
                break

    return x, y, r


def _polish_positions(x, y, r, w, h, time_limit=None):
    """Polish positions using force-directed method."""
    n = N
    start = time.time()
    dt = 0.0005

    for _ in range(500):
        if time_limit is not None and time.time() > time_limit:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 20
        fx += right_viol * 20
        fy -= bottom_viol * 20
        fy += top_viol * 20

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

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 8.0
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

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _slsqp_smooth(z, time_budget):
    """SLSQP with smooth penalty-based objective."""
    maxiter = int(min(400, time_budget * 60))
    if maxiter < 100:
        maxiter = 100

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    try:
        res = minimize(_smooth_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        return res.x
    except Exception:
        return None


def _slsqp_constraint_refine(z, time_budget):
    """SLSQP with explicit constraints for high-precision refinement."""
    n = N
    constraints = _build_constraints()
    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    maxiter = int(min(500, time_budget * 80))
    if maxiter < 50:
        maxiter = 50

    try:
        res = minimize(_constraint_objective, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
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
        if time.time() > deadline - 30:
            break

        # Force relax to resolve initial overlaps
        z_relaxed = _force_relax(z0, max_iter=150, time_limit=deadline - 30)

        # Smooth penalty-based SLSQP
        remaining = deadline - (time.time() - start_time) - 20
        if remaining > 5:
            z_opt = _slsqp_smooth(z_relaxed, remaining)
            if z_opt is not None:
                check_and_update(z_opt)

        # If we have a valid result, do constraint-based refinement
        if best_result is not None and time.time() < deadline - 25:
            remaining = deadline - (time.time() - start_time) - 15
            if remaining > 5:
                z_refined = _slsqp_constraint_refine(z_opt if z_opt is not None else z_relaxed, remaining)
                if z_refined is not None:
                    check_and_update(z_refined)

    # Phase 2: Polish the best result
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Position polish
        if time.time() < deadline - 8:
            x, y = _polish_positions(x, y, r, w, h, time_limit=deadline - 8)

        # Expansion relaxation polish (combined expand + local position search)
        if time.time() < deadline - 5:
            x, y, r = _expansion_relaxation_polish(x, y, r, w, h, time_limit=deadline - 5)

        # Reconstruct and check
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:N] = x
        z_new[1::3][:N] = y
        z_new[2::3][:N] = r
        z_new[-1] = w
        check_and_update(z_new)

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
