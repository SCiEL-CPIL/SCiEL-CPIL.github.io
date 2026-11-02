"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _soft_body_relax(z, max_iter=800, dt_init=0.003):
    """
    Soft-body elastic relaxation: circles can deform (temporary radius shrink)
    to squeeze past each other, then rigidify. This allows exploration of
    non-rigid intermediate states that rigid-body models cannot access.
    
    Each circle has a 'deformation' variable that allows it to temporarily
    reduce its effective radius, creating elastic repulsion forces that push
    circles into new configurations. After relaxation, radii are restored
    and a final rigid polish is applied.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return None

    # Deformation variables: how much each circle can shrink (fraction)
    deform = np.zeros(n)
    max_deform = 0.25  # max 25% radius shrinkage during soft phase
    deform_rate = 0.02  # how quickly deformation adjusts

    dt = dt_init
    iu = np.triu_indices(n, k=1)

    for it in range(max_iter):
        # Effective radii with deformation
        r_eff = r * (1.0 - deform)

        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Boundary forces (push away from walls)
        left_viol = np.maximum(r_eff - x, 0.0)
        right_viol = np.maximum(x + r_eff - w, 0.0)
        bottom_viol = np.maximum(r_eff - y, 0.0)
        top_viol = np.maximum(y + r_eff - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise elastic repulsion (soft-body: force ~ overlap^2 for smoothness)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum_eff = r_eff[:, None] + r_eff[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum_eff[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            # Elastic force: proportional to overlap (Hookean spring)
            f_mag = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                ii = iu[0][idx]
                jj = iu[1][idx]
                fx[ii] += ux[idx] * f_mag[idx]
                fy[ii] += uy[idx] * f_mag[idx]
                fx[jj] -= ux[idx] * f_mag[idx]
                fy[jj] -= uy[idx] * f_mag[idx]

        # Update deformation: circles that are heavily overlapped deform more
        # Deformation is driven by local crowding
        local_crowd = np.zeros(n)
        for idx in range(len(iu[0])):
            ii = iu[0][idx]
            jj = iu[1][idx]
            ov = overlap[idx]
            if ov > 0:
                local_crowd[ii] += ov
                local_crowd[jj] += ov

        # Adjust deformation: increase where crowded, decrease where free
        target_deform = np.clip(local_crowd * 0.5, 0.0, max_deform)
        deform += deform_rate * (target_deform - deform)
        deform = np.clip(deform, 0.0, max_deform)

        # Adaptive time step
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(0.0001, dt * 0.7)
        elif force_mag < 0.05:
            dt = min(0.008, dt * 1.15)

        # Update positions
        x += dt * fx
        y += dt * fy

        # Clip to rectangle (using effective radii)
        x = np.clip(x, r_eff, w - r_eff)
        y = np.clip(y, r_eff, h - r_eff)

        # Check convergence
        if force_mag < 0.005 and np.max(deform) < 0.01:
            break

    # Rigidify: restore original radii and do a quick rigid polish
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    # Quick rigid force-directed polish
    for it in range(200):
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

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_mag = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            for idx in range(len(iu[0])):
                ii = iu[0][idx]
                jj = iu[1][idx]
                fx[ii] += ux[idx] * f_mag[idx]
                fy[ii] += uy[idx] * f_mag[idx]
                fx[jj] -= ux[idx] * f_mag[idx]
                fy[jj] -= uy[idx] * f_mag[idx]
        else:
            break

        x += 0.002 * fx
        y += 0.002 * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
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

        # Hexagonal packing initial guesses (denser than square grid)
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.7]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            rows_needed = int(np.ceil(N / 6)) + 1
            col_spacing = w0 / 6.0
            row_spacing = col_spacing * np.sqrt(3) / 2.0
            idx = 0
            for row in range(rows_needed):
                if idx >= N:
                    break
                y_pos = (row + 0.5) * row_spacing
                if y_pos > h0 - 0.05:
                    break
                offset = (col_spacing / 2.0) if (row % 2 == 1) else 0.0
                for col in range(6):
                    if idx >= N:
                        break
                    x_pos = (col + 0.5) * col_spacing + offset
                    if x_pos < 0.05 or x_pos > w0 - 0.05:
                        continue
                    r0 = min(col_spacing, row_spacing) * 0.48
                    z[3 * idx] = x_pos
                    z[3 * idx + 1] = y_pos
                    z[3 * idx + 2] = r0
                    idx += 1
            if idx < N:
                for i in range(idx, N):
                    z[3 * i] = w0 / 2.0
                    z[3 * i + 1] = h0 / 2.0
                    z[3 * i + 2] = 0.05
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(12):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.14)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Try both rigid and soft-body relaxation
        z0_relaxed = _force_relax_unified(z0, max_iter=300)

        # Also try soft-body relaxation (allows squeezing through narrow gaps)
        z0_soft = _soft_body_relax(z0, max_iter=400, dt_init=0.002)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(120, remaining * 40)))

        # Try SLSQP from both starting points
        for z_start in [z0_relaxed, z0_soft]:
            if z_start is None:
                continue
            if time.time() > deadline - 25:
                break
            try:
                res = minimize(objective, z_start, method='SLSQP',
                               bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

                z_opt = res.x
                valid, val = check_validity(z_opt)

                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
                elif not valid:
                    z_repair = z_opt.copy()
                    r_idx = np.arange(2, len(z_repair) - 1, 3)
                    z_repair[r_idx] *= 0.95
                    valid, val = check_validity(z_repair)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_repair.copy()

            except Exception:
                continue

        # If soft-body gave a valid result, try to use it directly
        if z0_soft is not None:
            valid, val = check_validity(z0_soft)
            if valid and val > best_value:
                best_value = val
                best_result = z0_soft.copy()

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 20:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 12)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 14:
            expanded = _expansion_relaxation_polish_v2(best_result, deadline - (time.time() - start_time) - 6)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # Soft-body escape: try to find a better configuration via deformation
        if best_result is not None and time.time() < deadline - 12:
            soft_result = _soft_body_relax(best_result, max_iter=600, dt_init=0.001)
            if soft_result is not None:
                valid, val = check_validity(soft_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = soft_result

        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        if best_result is not None and time.time() < deadline - 5:
            smoothed = _smooth_epigraph_polish(best_result, deadline - (time.time() - start_time) - 1)
            if smoothed is not None:
                valid, val = check_validity(smoothed)
                if valid and val > best_value:
                    best_value = val
                    best_result = smoothed

        # Final high-precision Newton-style polish
        if best_result is not None and time.time() < deadline - 3:
            final_polished = _high_precision_polish(best_result, deadline - (time.time() - start_time))
            if final_polished is not None:
                valid, val = check_validity(final_polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_polished

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
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
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(5000, time_budget * 250))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

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
            f_pairs = overlap * 18.0
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

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    grow_factor = 1.0005
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish_v2(z, time_budget):
    """
    Vectorized expansion polish: computes max radii for all circles simultaneously,
    then performs coordinated position shifts.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)

    max_rounds = 60
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        np.fill_diagonal(d_all, np.inf)

        r_from_others = d_all - r[None, :]
        np.fill_diagonal(r_from_others, np.inf)
        r_from_neigh = np.min(r_from_others, axis=1)

        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        r_max = np.maximum(0.0, np.minimum(r_wall, r_from_neigh))

        grow_mask = r_max > r + 1e-7
        if np.any(grow_mask):
            r_new = np.where(grow_mask, np.minimum(r_max * 0.95, r + (r_max - r) * 0.5), r)
            if _is_valid(x, y, r_new, w, h):
                r = r_new
            else:
                for i in np.where(grow_mask)[0]:
                    r_trial = r.copy()
                    r_trial[i] = r_max[i] * 0.95
                    if _is_valid(x, y, r_trial, w, h):
                        r = r_trial

        margin_left = x - r
        margin_right = w - x - r
        margin_bottom = y - r
        margin_top = h - y - r

        d_pairs = d_all[iu]
        r_sum_pairs = r[iu[0]] + r[iu[1]]
        gap_pairs = d_pairs - r_sum_pairs

        min_gap = np.full(n, 0.05)
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            g = gap_pairs[idx]
            if g < min_gap[i_idx]:
                min_gap[i_idx] = g
            if g < min_gap[j_idx]:
                min_gap[j_idx] = g

        min_margin = np.minimum(np.minimum(margin_left, margin_right),
                                np.minimum(margin_bottom, margin_top))
        min_constraint = np.minimum(min_gap, min_margin)

        stuck = min_constraint < 0.015
        if not np.any(stuck):
            break

        improved = False
        for i in np.where(stuck)[0]:
            ml = margin_left[i]
            mr = margin_right[i]
            mb = margin_bottom[i]
            mt = margin_top[i]
            mg = min_gap[i]

            move_x = 0.0
            move_y = 0.0

            if mg <= ml and mg <= mr and mg <= mb and mg <= mt:
                best_j = -1
                best_d = float('inf')
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    dx = x[i] - x[best_j]
                    dy = y[i] - y[best_j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-8:
                        move_x = dx / d
                        move_y = dy / d
            elif ml <= mr and ml <= mb and ml <= mt:
                move_x = 1.0
            elif mr <= ml and mr <= mb and mr <= mt:
                move_x = -1.0
            elif mb <= ml and mb <= mr and mb <= mt:
                move_y = 1.0
            else:
                move_y = -1.0

            for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                new_x = x[i] + move_x * step
                new_y = y[i] + move_y * step
                if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                    x[i] = new_x
                    y[i] = new_y
                    improved = True
                    break
            else:
                perp_x = -move_y
                perp_y = move_x
                for step in [0.02, 0.01, 0.005]:
                    for sign in [1.0, -1.0]:
                        new_x = x[i] + sign * perp_x * step
                        new_y = y[i] + sign * perp_y * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            break
                    if improved:
                        break

        if not improved:
            break

    grow_factor = 1.001
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
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


def _smooth_epigraph_polish(z, time_budget):
    """
    Smooth epigraph formulation using adaptive log-sum-exp soft-max for violations.
    Uses annealing on the log-sum-exp temperature to first find feasible regions
    then tighten toward exact constraints.
    """
    z = z.copy().astype(float)
    n = N

    def smooth_objective(zv, beta=30.0):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0.001):
            return 1e6

        obj = -np.sum(r)

        # Pairwise violations
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        raw_overlap = r_sum_pairs - d_pairs
        positive_overlap = np.maximum(raw_overlap, 0.0)

        max_pos = np.max(positive_overlap) if positive_overlap.size > 0 else 0.0
        if max_pos > 1e-15:
            shifted = beta * (positive_overlap - max_pos)
            # Numerically stable log-sum-exp
            max_shifted = np.max(shifted)
            lse = max_pos + (max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))) / beta
            obj += 800.0 * lse
        else:
            obj += 800.0 * np.sum(positive_overlap * positive_overlap)

        # Boundary violations
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
        max_viol = np.max(all_viol) if all_viol.size > 0 else 0.0
        if max_viol > 1e-15:
            shifted_b = beta * (all_viol - max_viol)
            max_shifted_b = np.max(shifted_b)
            lse_b = max_viol + (max_shifted_b + np.log(np.sum(np.exp(shifted_b - max_shifted_b)))) / beta
            obj += 800.0 * lse_b
        else:
            obj += 800.0 * np.sum(all_viol * all_viol)

        return obj

    bounds = [(0.01, 1.99)] * (2 * n) + [(0.005, 0.5)] * n + [(0.4, 1.6)]

    # Annealing: start with low beta (smooth), increase to high beta (sharp)
    betas = [15.0, 30.0, 60.0, 100.0]
    z_current = z.copy()
    remaining_time = time_budget

    for bi, beta in enumerate(betas):
        if time.time() > time_budget - 1:
            break
        seg_time = remaining_time / (len(betas) - bi)
        maxiter = int(min(300, seg_time * 100))
        if maxiter < 30:
            maxiter = 30

        try:
            res = minimize(lambda zv: smooth_objective(zv, beta), z_current, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            z_opt = res.x
            x = z_opt[0::3][:n]
            y = z_opt[1::3][:n]
            r = z_opt[2::3][:n]
            w = z_opt[-1]
            h = 2.0 - w

            if _is_valid(x, y, r, w, h):
                z_current = z_opt
            elif res.fun < smooth_objective(z_current, beta):
                z_current = z_opt

        except Exception:
            break

    x = z_current[0::3][:n]
    y = z_current[1::3][:n]
    r = z_current[2::3][:n]
    w = z_current[-1]
    h = 2.0 - w

    if _is_valid(x, y, r, w, h):
        return z_current
    return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: (
                (p[3 * i] - p[3 * j]) ** 2 +
                (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                (p[3 * i + 2] + p[3 * j + 2]) ** 2
            )})

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

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


def _high_precision_polish(z, time_budget):
    """
    Final high-precision polish using iterative coordinate descent with
    exact radius maximization and Newton-style position refinement.
    Targets the last few thousandths of radius sum.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    max_rounds = int(min(200, time_budget * 30))

    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False

        # Step 1: Try to grow each circle's radius maximally
        for i in range(n):
            # Compute max possible radius for circle i given current positions
            # Limited by walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            # Limited by neighbors
            r_neigh = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-10:
                    r_avail = d - r[j]
                    if r_avail < r_neigh:
                        r_neigh = r_avail
            r_max_i = max(0.0, min(r_wall, r_neigh))

            if r_max_i > r[i] + 1e-10:
                # Try to grow
                r_trial = r.copy()
                r_trial[i] = min(r_max_i * 0.999, r[i] + (r_max_i - r[i]) * 0.8)
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    improved = True

        # Step 2: Coordinate descent - try small moves for each circle
        for i in range(n):
            if time.time() > time_budget:
                break

            # Determine best direction to move circle i
            # Compute constraint tightness
            margin_l = x[i] - r[i]
            margin_r = w - x[i] - r[i]
            margin_b = y[i] - r[i]
            margin_t = h - y[i] - r[i]

            # Find nearest neighbor gap
            min_gap = 0.1
            nearest_dir = (0.0, 0.0)
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - r[i] - r[j]
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-10:
                        nearest_dir = (dx / d, dy / d)

            # Try moving in several directions
            directions = []
            if min_gap < 0.02:
                # Move away from nearest neighbor
                directions.append((nearest_dir[0], nearest_dir[1]))
                # Perpendicular
                directions.append((-nearest_dir[1], nearest_dir[0]))
                directions.append((nearest_dir[1], -nearest_dir[0]))

            # Wall directions
            if margin_l < 0.02:
                directions.append((1.0, 0.0))
            if margin_r < 0.02:
                directions.append((-1.0, 0.0))
            if margin_b < 0.02:
                directions.append((0.0, 1.0))
            if margin_t < 0.02:
                directions.append((0.0, -1.0))

            # Diagonal directions
            directions.extend([(0.707, 0.707), (0.707, -0.707),
                              (-0.707, 0.707), (-0.707, -0.707)])

            for dx_dir, dy_dir in directions:
                for step in [0.03, 0.015, 0.008, 0.004, 0.002, 0.001]:
                    new_x = x[i] + dx_dir * step
                    new_y = y[i] + dy_dir * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                if improved:
                    break

        # Step 3: Try to grow all radii slightly
        if improved:
            r_trial = r * 1.0003
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
            else:
                r_trial = r * 1.0001
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial

        if not improved:
            break

    # Final radius growth sweep
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
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
