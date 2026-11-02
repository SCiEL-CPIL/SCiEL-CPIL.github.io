"""Circle packing in perimeter-4 rectangle: SLSQP + Soft-Body Physics Polish."""
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

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
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


def _smooth_objective(z, penalty_weight=500.0, beta=50.0):
    """Log-sum-exp smooth objective for better gradient landscape."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise: log-sum-exp of (r_i + r_j - d_ij)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

    # Smooth max via log-sum-exp: log(sum(exp(beta * val))) / beta
    overlap_vals = r_sum_pairs - d_pairs
    # Clip to avoid overflow
    clipped = np.clip(overlap_vals, -50.0, 50.0)
    max_val = np.max(clipped)
    if max_val > 0:
        lse = max_val + np.log(np.sum(np.exp(beta * (clipped - max_val)))) / beta
        overlap_penalty = penalty_weight * lse
    else:
        overlap_penalty = 0.0
    obj += overlap_penalty

    # Boundary: smooth sum of violations
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    clipped_v = np.clip(all_viol, -50.0, 50.0)
    max_v = np.max(clipped_v)
    if max_v > 0:
        lse_v = max_v + np.log(np.sum(np.exp(beta * (clipped_v - max_v)))) / beta
        boundary_penalty = penalty_weight * lse_v
    else:
        boundary_penalty = 0.0
    obj += boundary_penalty

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _epigraph_radius_growth(z, time_budget):
    """Epigraph formulation: introduce auxiliary variable t >= max constraint violation,
    minimize -sum(r) + penalty*t. This gives a smoother optimization landscape."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Iteratively grow radii using a smooth feasibility check
    # and local position adjustments
    max_rounds = int(min(50, time_budget * 3))
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        # Try to grow each circle
        for i in range(n):
            # Compute max possible radius for circle i
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

            if r_max > r[i] + 1e-8:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-9):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

        if not improved:
            # Try small position perturbations to unlock more growth
            rng_local = np.random.default_rng(rnd * 1000 + 42)
            for i in range(n):
                for _ in range(5):
                    if time.time() > time_budget:
                        break
                    dx = rng_local.uniform(-0.01, 0.01)
                    dy = rng_local.uniform(-0.01, 0.01)
                    new_x = x[i] + dx
                    new_y = y[i] + dy
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        # Check if this position allows more growth
                        r_wall = min(new_x, w - new_x, new_y, h - new_y)
                        r_max_new = r_wall
                        for j in range(n):
                            if i == j:
                                continue
                            ddx = new_x - x[j]
                            ddy = new_y - y[j]
                            d = np.sqrt(ddx * ddx + ddy * ddy)
                            r_allowed = d - r[j]
                            if r_allowed < r_max_new:
                                r_max_new = r_allowed
                        r_max_new = max(0.0, r_max_new)

                        # Current max radius at current position
                        r_wall_cur = min(x[i], w - x[i], y[i], h - y[i])
                        r_max_cur = r_wall_cur
                        for j in range(n):
                            if i == j:
                                continue
                            ddx = x[i] - x[j]
                            ddy = y[i] - y[j]
                            d = np.sqrt(ddx * ddx + ddy * ddy)
                            r_allowed = d - r[j]
                            if r_allowed < r_max_cur:
                                r_max_cur = r_allowed
                        r_max_cur = max(0.0, r_max_cur)

                        if r_max_new > r_max_cur + 1e-7:
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            break

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
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    # Generate diverse initial guesses - Optimized for time efficiency
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # Focused set of aspect ratios to balance exploration and time
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
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

        # Random starts - slightly more than inspiration to cover noise
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

        # One hexagonal guess for structural diversity
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        spacing_x = w0 / (cols + 0.5)
        spacing_y = h0 / (rows + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            cy = (row + 0.5) * spacing_y
            r0 = min(spacing_x, spacing_y) * 0.48
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Smooth objective for L-BFGS-B (better for smooth landscapes)
    def smooth_obj(z):
        return _smooth_objective(z, penalty_weight=500.0, beta=50.0)

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

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

        # Second pass: L-BFGS-B with smooth log-sum-exp objective
        if time.time() > deadline - 25:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        try:
            res2 = minimize(smooth_obj, z0_relaxed, method='L-BFGS-B',
                            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                            options={'maxiter': int(min(300, remaining * 40)),
                                     'ftol': 1e-14, 'maxls': 50})
            z_opt2 = res2.x
            valid2, val2 = check_validity(z_opt2)
            if valid2 and val2 > best_value:
                best_value = val2
                best_result = z_opt2.copy()
        except Exception:
            continue

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Epigraph-based radius growth
        if best_result is not None and time.time() < deadline - 20:
            epigraph = _epigraph_radius_growth(best_result, deadline - (time.time() - start_time) - 18)
            if epigraph is not None:
                valid, val = check_validity(epigraph)
                if valid and val > best_value:
                    best_value = val
                    best_result = epigraph

        # 3. Expansion/Relocation Polish - Strong local improvement
        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 12)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 7)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final high-precision polish with more time budget
        if best_result is not None and time.time() < deadline - 4:
            polished2 = _high_precision_polish(best_result, deadline - (time.time() - start_time) - 2)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

        # 6. Second pass of high-precision polish if time allows
        if best_result is not None and time.time() < deadline - 1:
            polished3 = _high_precision_polish(best_result, deadline - (time.time() - start_time))
            if polished3 is not None:
                valid, val = check_validity(polished3)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished3

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


def _force_relax_unified(z, max_iter=500):
    """Soft-body physics relaxation: circles act as deformable blobs with elastic repulsion.
    
    Uses a smooth potential energy model where:
    1. Overlap forces follow Hooke's law (linear in penetration depth)
    2. A "softness" parameter allows temporary deformation
    3. Damping prevents oscillation
    4. Adaptive stiffness increases as we converge to rigid configuration
    """
    z = z.copy().astype(float)
    n = N
    dt = 0.003
    min_dt = 0.00005
    convergence_tol = 1e-9
    damping = 0.85
    
    # Soft-body parameters
    base_stiffness = 20.0
    softness = 0.3  # Fraction of overlap that can be "absorbed" by deformation
    
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    for it in range(max_iter):
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion with smooth potential (quadratic for smoothness)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        wall_stiffness = 150.0
        fx -= left_viol * wall_stiffness
        fx += right_viol * wall_stiffness
        fy -= bottom_viol * wall_stiffness
        fy += top_viol * wall_stiffness

        # Pairwise elastic repulsion (soft-body model)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        raw_overlap = r_sum_pairs - d_pairs
        max_overlap = float(np.max(raw_overlap)) if raw_overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            # Soft-body: only penalize overlap beyond the "deformation allowance"
            # This allows circles to temporarily deform and squeeze past each other
            effective_overlap = np.maximum(raw_overlap - softness * r_sum_pairs, 0.0)
            
            # Adaptive stiffness: increases as we get closer to rigid configuration
            progress = min(1.0, max_overlap / 0.1)  # 1 when far, 0 when close
            stiffness = base_stiffness * (0.5 + 0.5 * progress)
            
            f_pairs = effective_overlap * stiffness
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

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Update velocities with damping (soft-body dynamics)
        vx = vx * damping + dt * fx
        vy = vy * damping + dt * fy

        # Clamp velocities to prevent explosion
        max_vel = 0.05
        vx = np.clip(vx, -max_vel, max_vel)
        vy = np.clip(vy, -max_vel, max_vel)

        # Update positions
        x += vx
        y += vy

        # Clamp positions within bounds
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        z[0::3][:n] = x
        z[1::3][:n] = y

    return z


def _polish_result(z, time_budget):
    """Polish with soft-body dynamics: circles deform elastically to find better packings."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0008
    max_iter = int(min(5000, time_budget * 200))
    damping = 0.9
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    # Soft-body parameters for polishing
    base_stiffness = 25.0
    softness = 0.15  # Less softness than initial relaxation
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        wall_stiffness = 200.0
        fx -= left_viol * wall_stiffness
        fx += right_viol * wall_stiffness
        fy -= bottom_viol * wall_stiffness
        fy += top_viol * wall_stiffness

        # Pairwise elastic repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        raw_overlap = r_sum_pairs - d_pairs
        max_overlap = float(np.max(raw_overlap)) if raw_overlap.size > 0 else 0.0

        if max_overlap > 1e-10:
            effective_overlap = np.maximum(raw_overlap - softness * r_sum_pairs, 0.0)
            progress = min(1.0, max_overlap / 0.05)
            stiffness = base_stiffness * (0.5 + 0.5 * progress)
            
            f_pairs = effective_overlap * stiffness
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

        # Update velocities with damping
        vx = vx * damping + dt * fx
        vy = vy * damping + dt * fy

        max_vel = 0.02
        vx = np.clip(vx, -max_vel, max_vel)
        vy = np.clip(vy, -max_vel, max_vel)

        x += vx
        y += vy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
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


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    Improved: multi-step moves, binary search for max radius, more rounds.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            # 1. Try to expand radius i to max possible using binary search
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
            
            if r_max > r[i] + 1e-6:
                # Binary search for the true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True
            
            # 2. If expansion is limited, try to shift circle i to create space
            min_dist = float('inf')
            blocker = None
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]
            
            if d_left < min_dist:
                min_dist = d_left
                blocker = 'left'
            if d_right < min_dist:
                min_dist = d_right
                blocker = 'right'
            if d_bottom < min_dist:
                min_dist = d_bottom
                blocker = 'bottom'
            if d_top < min_dist:
                min_dist = d_top
                blocker = 'top'
            
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    blocker = j
            
            # If touching a constraint, try multi-step moves away
            if min_dist < 0.02:
                move_x = 0.0
                move_y = 0.0
                if blocker == 'left':
                    move_x = 1.0
                elif blocker == 'right':
                    move_x = -1.0
                elif blocker == 'bottom':
                    move_y = 1.0
                elif blocker == 'top':
                    move_y = -1.0
                else:
                    j = blocker
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        move_x = dx / d
                        move_y = dy / d
                
                # Try multiple step sizes, from larger to smaller
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                else:
                    # Try perpendicular directions too
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005]:
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

    # Final uniform expansion with binary search
    grow_factor = 1.001
    for _ in range(100):
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

def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
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
        if i == j: continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx*dx + dy*dy
        r_sum = r[i] + r[j]
        if d2 < r_sum*r_sum - 1e-9:
            return False
    return True


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

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

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
        h = z_opt[-1] if False else 2.0 - z_opt[-1] # Correct h calculation

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


def _high_precision_polish(z, time_budget):
    """
    High-precision polish with soft-body dynamics for final squeeze.
    Uses very tight tolerances and elastic deformation to maximize radii.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Soft-body fine-tuning with very small steps
    dt = 0.0002
    max_iter = int(min(10000, time_budget * 400))
    damping = 0.92
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    # Very stiff for high-precision polish
    base_stiffness = 50.0
    softness = 0.05  # Minimal deformation allowed
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 300
        fx += right_viol * 300
        fy -= bottom_viol * 300
        fy += top_viol * 300
        
        # Pairwise elastic repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        raw_overlap = r_sum_pairs - d_pairs
        max_overlap = float(np.max(raw_overlap)) if raw_overlap.size > 0 else 0.0
        
        if max_overlap > 1e-12:
            effective_overlap = np.maximum(raw_overlap - softness * r_sum_pairs, 0.0)
            progress = min(1.0, max_overlap / 0.02)
            stiffness = base_stiffness * (0.5 + 0.5 * progress)
            
            f_pairs = effective_overlap * stiffness
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
        
        # Update with damping
        vx = vx * damping + dt * fx
        vy = vy * damping + dt * fy
        max_vel = 0.005
        vx = np.clip(vx, -max_vel, max_vel)
        vy = np.clip(vy, -max_vel, max_vel)
        
        x += vx
        y += vy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    
    # Phase 2: Local position search for each circle
    for _ in range(25):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max_current = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max_current:
                    r_max_current = r_allowed
            r_max_current = max(0.0, r_max_current)
            
            best_r_max = r_max_current
            best_pos = (x[i], y[i])
            
            for step in [0.005, 0.002, 0.001, 0.0005]:
                for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                    new_x = x[i] + step * np.cos(angle)
                    new_y = y[i] + step * np.sin(angle)
                    
                    if new_x < r[i] or new_x > w - r[i]:
                        continue
                    if new_y < r[i] or new_y > h - r[i]:
                        continue
                    
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d2 = dx*dx + dy*dy
                        r_sum = r[i] + r[j]
                        if d2 < r_sum*r_sum - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    r_wall_new = min(new_x, w - new_x, new_y, h - new_y)
                    r_max_new = r_wall_new
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        r_allowed = d - r[j]
                        if r_allowed < r_max_new:
                            r_max_new = r_allowed
                    r_max_new = max(0.0, r_max_new)
                    
                    if r_max_new > best_r_max + 1e-7:
                        best_r_max = r_max_new
                        best_pos = (new_x, new_y)
            
            if best_pos != (x[i], y[i]):
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                improved = True
        
        if not improved:
            break
    
    # Phase 3: Greedy radius expansion with binary search
    for _ in range(60):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
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
                lo = r[i]
                hi = r_max
                for _ in range(35):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True
        
        if not improved:
            break
    
    # Phase 4: Uniform expansion with very fine steps
    for _ in range(400):
        if time.time() > time_budget:
            break
        r_trial = r * 1.00003
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break
    
    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None
