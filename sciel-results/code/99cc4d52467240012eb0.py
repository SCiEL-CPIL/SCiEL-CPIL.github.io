"""Circle packing in perimeter-4 rectangle: SLSQP + Smooth Epigraph + Advanced Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_NUM_PAIRS = N * (N - 1) // 2
_IU = np.triu_indices(N, k=1)


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

    # Vectorized pairwise overlap penalty (smooth via soft-min)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    d_sq_pairs = d_sq[_IU]
    r_sum_pairs = r_sum[_IU]

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


def _smooth_objective(z, beta=50.0):
    """
    Smooth log-sum-exp based objective. Uses soft-max to create a differentiable
    surrogate for the max-overlap constraint, enabling better gradient flow.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Smooth pairwise overlap: use log-sum-exp of (r_i + r_j - d_ij)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    d_sq_pairs = d_sq[_IU]
    r_sum_pairs = r_sum[_IU]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

    # overlap_ij = r_i + r_j - d_ij (positive when overlapping)
    overlap_vals = r_sum_pairs - d_pairs

    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * v)))
    # This gives a smooth approximation to max(overlap_vals, 0)
    shifted = beta * overlap_vals
    shifted = np.clip(shifted, -500, 500)  # numerical safety
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_max_overlap = lse / beta

    # Also penalize individual overlaps with quadratic for local structure
    hard_overlap = np.maximum(overlap_vals, 0.0)
    quad_penalty = 100.0 * np.sum(hard_overlap * hard_overlap)

    obj += 500.0 * smooth_max_overlap + quad_penalty

    # Smooth boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    shifted_b = beta * all_viol
    shifted_b = np.clip(shifted_b, -500, 500)
    max_b = np.max(shifted_b)
    lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
    smooth_max_viol = lse_b / beta

    hard_viol = np.maximum(all_viol, 0.0)
    obj += 500.0 * smooth_max_viol + 100.0 * np.sum(hard_viol * hard_viol)

    # Aspect ratio penalty (smooth)
    if w < 0.5:
        obj += 200.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 200.0 * (w - 1.5) ** 2

    return obj


def _epigraph_radius_growth(x, y, r, w, h, time_budget):
    """
    Smooth epigraph-based radius growth with smooth max-flow style formulation.
    Uses auxiliary variable t to represent minimum clearance, with smooth
    log-sum-exp surrogate for the constraint violations.
    """
    n = N
    iu = _IU
    
    def epigraph_obj(v):
        xi = v[0:n]
        yi = v[n:2*n]
        ri = v[2*n:3*n]
        t = v[3*n]
        wi = v[3*n+1]
        hi = 2.0 - wi
        
        if wi < 0.2 or hi < 0.2 or np.any(ri < 0) or t < -0.1:
            return 1e6
        
        # Objective: maximize sum(r) + lambda * t (clearance)
        obj = -np.sum(ri) - 5.0 * t
        
        # Pairwise: d_ij >= ri + rj + t
        dx = xi[:, None] - xi[None, :]
        dy = yi[:, None] - yi[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = ri[:, None] + ri[None, :]
        
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        # Violation: (ri + rj + t) - d_ij (positive when violated)
        viol = r_sum_pairs + t - d_pairs
        
        # Smooth max via log-sum-exp for the worst violation
        beta = 30.0
        shifted = beta * viol
        shifted = np.clip(shifted, -500, 500)
        max_shifted = np.max(shifted)
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        smooth_max_viol = lse / beta
        
        # Combine: smooth max + quadratic for local structure
        hard_viol = np.maximum(viol, 0.0)
        obj += 200.0 * smooth_max_viol + 50.0 * np.sum(hard_viol * hard_viol)
        
        # Boundary constraints with same smooth treatment
        left_viol = ri - xi
        right_viol = xi + ri - wi
        bottom_viol = ri - yi
        top_viol = yi + ri - hi
        
        all_bviol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
        shifted_b = beta * all_bviol
        shifted_b = np.clip(shifted_b, -500, 500)
        max_b = np.max(shifted_b)
        lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
        smooth_max_bviol = lse_b / beta
        
        hard_bviol = np.maximum(all_bviol, 0.0)
        obj += 200.0 * smooth_max_bviol + 50.0 * np.sum(hard_bviol * hard_bviol)
        
        return obj
    
    # Initial guess
    v0 = np.zeros(3*n + 2)
    v0[0:n] = x
    v0[n:2*n] = y
    v0[2*n:3*n] = r
    v0[3*n] = 0.0  # t starts at 0
    v0[3*n+1] = w
    
    bounds = [(0, 2)] * n + [(0, 2)] * n + [(0.001, 0.5)] * n + [(-0.05, 0.15)] + [(0.3, 1.7)]
    
    maxiter = int(min(600, time_budget * 120))
    try:
        res = minimize(epigraph_obj, v0, method='SLSQP', bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 100})
        v_opt = res.x
        xi_opt = v_opt[0:n]
        yi_opt = v_opt[n:2*n]
        ri_opt = v_opt[2*n:3*n]
        w_opt = v_opt[3*n+1]
        h_opt = 2.0 - w_opt
        
        if _is_valid(xi_opt, yi_opt, ri_opt, w_opt, h_opt, tol=1e-8):
            z_new = np.empty(3*n + 1)
            z_new[0::3][:n] = xi_opt
            z_new[1::3][:n] = yi_opt
            z_new[2::3][:n] = ri_opt
            z_new[-1] = w_opt
            return z_new
    except Exception:
        pass
    
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

        # Add a "staggered grid" guess with tighter packing
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        sx = w0 / (cols + 0.3)
        sy = h0 / (rows + 0.3)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * sx
            cy = (row + 0.5) * sy
            r0 = min(sx, sy) * 0.49
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        # Add a "corner-biased" guess: place larger circles at corners
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        # 4 corners
        corner_positions = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for k, (cx, cy) in enumerate(corner_positions):
            z[3 * k] = cx * w0
            z[3 * k + 1] = cy * h0
            z[3 * k + 2] = 0.12
        # Remaining 17 in a grid in the center
        remaining = N - 4
        r_side = int(np.ceil(np.sqrt(remaining)))
        cx0, cy0 = 0.3, 0.3
        cw, ch = 0.4, 0.4
        for k in range(remaining):
            row, col = divmod(k, r_side)
            cx = (cx0 + (col + 0.5) / r_side * cw) * w0
            cy = (cy0 + (row + 0.5) / r_side * ch) * h0
            r0 = min(cw / r_side, ch / r_side) * 0.45
            idx = 4 + k
            z[3 * idx] = cx
            z[3 * idx + 1] = cy
            z[3 * idx + 2] = r0
        z[-1] = w0
        guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization with penalty objective
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
            pass

        # Second pass: try smooth log-sum-exp objective from same start
        if time.time() < deadline - 25:
            try:
                res2 = minimize(_smooth_objective, z0_relaxed, method='SLSQP',
                                bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                                options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 60})
                z_opt2 = res2.x
                valid2, val2 = check_validity(z_opt2)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = z_opt2.copy()
            except Exception:
                pass

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish - Strong local improvement
        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 2b. Multi-start perturbation + expansion polish
        if best_result is not None and time.time() < deadline - 12:
            rng2 = np.random.default_rng(123)
            for trial in range(5):
                if time.time() > deadline - 12:
                    break
                z_pert = best_result.copy()
                # Small random perturbation to positions
                x_p = z_pert[0::3][:N] + rng2.normal(0, 0.01, N)
                y_p = z_pert[1::3][:N] + rng2.normal(0, 0.01, N)
                # Clamp
                w_p = z_pert[-1]
                h_p = 2.0 - w_p
                r_p = z_pert[2::3][:N]
                x_p = np.clip(x_p, r_p, w_p - r_p)
                y_p = np.clip(y_p, r_p, h_p - r_p)
                z_pert[0::3][:N] = x_p
                z_pert[1::3][:N] = y_p

                # Quick force relaxation
                z_pert = _force_relax_unified(z_pert, max_iter=200)

                # Expansion polish
                expanded2 = _expansion_relaxation_polish(z_pert, deadline - (time.time() - start_time) - 10)
                if expanded2 is not None:
                    valid2, val2 = check_validity(expanded2)
                    if valid2 and val2 > best_value:
                        best_value = val2
                        best_result = expanded2

        # 3. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 3b. NEW: Epigraph radius growth for smooth uniform expansion
        if best_result is not None and time.time() < deadline - 15:
            x_e = best_result[0::3][:N]
            y_e = best_result[1::3][:N]
            r_e = best_result[2::3][:N]
            w_e = best_result[-1]
            h_e = 2.0 - w_e
            epigraph_result = _epigraph_radius_growth(x_e, y_e, r_e, w_e, h_e, 
                                                       deadline - (time.time() - start_time) - 12)
            if epigraph_result is not None:
                valid_e, val_e = check_validity(epigraph_result)
                if valid_e and val_e > best_value:
                    best_value = val_e
                    best_result = epigraph_result

        # 3c. Second epigraph pass with different initial t
        if best_result is not None and time.time() < deadline - 12:
            x_e2 = best_result[0::3][:N]
            y_e2 = best_result[1::3][:N]
            r_e2 = best_result[2::3][:N]
            w_e2 = best_result[-1]
            h_e2 = 2.0 - w_e2
            # Slightly shrink radii to give more room, then epigraph will regrow
            r_e2_shrunk = r_e2 * 0.995
            epigraph_result2 = _epigraph_radius_growth(x_e2, y_e2, r_e2_shrunk, w_e2, h_e2, 
                                                       deadline - (time.time() - start_time) - 10)
            if epigraph_result2 is not None:
                valid_e2, val_e2 = check_validity(epigraph_result2)
                if valid_e2 and val_e2 > best_value:
                    best_value = val_e2
                    best_result = epigraph_result2

        # 4. Final high-precision polish
        if best_result is not None and time.time() < deadline - 3:
            polished2 = _high_precision_polish(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

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

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion (vectorized)
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

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
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
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        # Pairwise repulsion (vectorized)
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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
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
    """
    High-precision SLSQP refinement using explicit constraints.
    Improved: tighter ftol, more iterations, and a two-phase approach
    (first fix positions, then grow radii).
    """
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

    # Phase 1: Optimize with current bounds
    maxiter1 = int(min(600, time_budget * 100))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter1, 'ftol': 1e-14, 'maxls': 120})
        z_opt = res.x
    except Exception:
        return None

    x = z_opt[0::3][:n]
    y = z_opt[1::3][:n]
    r = z_opt[2::3][:n]
    w = z_opt[-1]
    h = 2.0 - z_opt[-1]

    if not _is_valid(x, y, r, w, h):
        # Try small shrink
        for shrink in [0.999, 0.998, 0.997]:
            r_shrunk = r * shrink
            if _is_valid(x, y, r_shrunk, w, h):
                z_opt[2::3][:n] = r_shrunk
                return z_opt
        return None

    # Phase 2: Try to further grow radii by tightening bounds on positions
    # and re-optimizing with a slightly larger radius lower bound
    if time.time() < time_budget - 5:
        r_min = float(np.min(r)) * 1.001
        bounds2 = []
        for i in range(n):
            bounds2.extend([(0, 2), (0, 2), (r_min, 1.0)])
        bounds2.append((0.3, 1.7))

        maxiter2 = int(min(400, (time_budget - 5 - (time.time() - (time_budget - time_budget))) * 80))
        maxiter2 = int(min(400, 300))
        try:
            res2 = minimize(obj_refine, z_opt, method='SLSQP',
                           bounds=bounds2,
                           constraints=constraints,
                           options={'maxiter': maxiter2, 'ftol': 1e-14, 'maxls': 120})
            z_opt2 = res2.x
            x2 = z_opt2[0::3][:n]
            y2 = z_opt2[1::3][:n]
            r2 = z_opt2[2::3][:n]
            w2 = z_opt2[-1]
            h2 = 2.0 - w2
            if _is_valid(x2, y2, r2, w2, h2) and np.sum(r2) > np.sum(r):
                return z_opt2
        except Exception:
            pass

    return z_opt


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
    High-precision polish with improved multi-phase strategy:
    1. Fine position relaxation
    2. Greedy individual expansion with position nudging
    3. Global uniform expansion via binary search
    4. Final micro-expansion
    5. Aspect ratio optimization: try small w adjustments to find better packing
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_val = float(np.sum(r))
    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

    # Phase 1: Fine-tune positions with very small steps
    dt = 0.0002
    max_iter = int(min(3000, time_budget * 150))
    
    for it in range(max_iter):
        if time.time() > time_budget - 10:
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
        
        d_sq_pairs = d_sq[_IU]
        r_sum_pairs = r_sum[_IU]
        
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 25.0
            dx_pairs = dx[_IU]
            dy_pairs = dy[_IU]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            
            for idx in range(len(_IU[0])):
                i_idx = _IU[0][idx]
                j_idx = _IU[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]
        
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    
    # Phase 2: Alternating individual expansion + micro position adjustment
    for round_num in range(60):
        if time.time() > time_budget - 10:
            break
        
        improved = False
        for i in range(n):
            # Calculate max possible radius given current positions
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
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(22):
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
            
            # After expanding, try to nudge circle i slightly to create more room
            if improved:
                d_left = x[i] - r[i]
                d_right = w - x[i] - r[i]
                d_bot = y[i] - r[i]
                d_top = h - y[i] - r[i]
                min_d = min(d_left, d_right, d_bot, d_top)
                dx_move = 0.0
                dy_move = 0.0
                if d_left == min_d:
                    dx_move = 0.004
                elif d_right == min_d:
                    dx_move = -0.004
                elif d_bot == min_d:
                    dy_move = 0.004
                elif d_top == min_d:
                    dy_move = -0.004
                
                if dx_move != 0 or dy_move != 0:
                    new_xi = x[i] + dx_move
                    new_yi = y[i] + dy_move
                    if _is_valid_move(x, y, r, w, h, i, new_xi, new_yi):
                        x[i] = new_xi
                        y[i] = new_yi
    
    # Phase 3: Uniform expansion via binary search
    lo_factor = 1.0
    hi_factor = 1.03
    for _ in range(35):
        if time.time() > time_budget - 5:
            break
        mid_factor = (lo_factor + hi_factor) / 2.0
        r_trial = r * mid_factor
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            lo_factor = mid_factor
            r = r_trial
        else:
            hi_factor = mid_factor
    
    # Phase 4: Final micro-expansion
    for _ in range(80):
        if time.time() > time_budget - 3:
            break
        r_trial = r * 1.00005
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break
    
    # Track best so far
    if _is_valid(x, y, r, w, h, tol=1e-10) and np.sum(r) > best_val:
        best_val = float(np.sum(r))
        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

    # Phase 5: Aspect ratio optimization - try small w adjustments
    if time.time() < time_budget - 5:
        for dw in [-0.02, -0.01, 0.01, 0.02, -0.03, 0.03]:
            if time.time() > time_budget - 3:
                break
            w_try = w + dw
            h_try = 2.0 - w_try
            if w_try < 0.3 or h_try < 0.3:
                continue
            
            # Scale positions to fit new aspect ratio
            x_try = x * (w_try / w)
            y_try = y * (h_try / h)
            r_try = r.copy()
            
            # Check if still valid after scaling
            if _is_valid(x_try, y_try, r_try, w_try, h_try, tol=1e-8):
                # Try to expand radii in new aspect ratio
                for _ in range(30):
                    if time.time() > time_budget - 3:
                        break
                    r_grow = r_try * 1.001
                    if _is_valid(x_try, y_try, r_grow, w_try, h_try, tol=1e-9):
                        r_try = r_grow
                    else:
                        break
                
                if _is_valid(x_try, y_try, r_try, w_try, h_try, tol=1e-9) and np.sum(r_try) > best_val:
                    best_val = float(np.sum(r_try))
                    best_x, best_y, best_r, best_w = x_try.copy(), y_try.copy(), r_try.copy(), w_try

    if _is_valid(best_x, best_y, best_r, best_w, 2.0 - best_w, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = best_w
        return z_new
    return None
