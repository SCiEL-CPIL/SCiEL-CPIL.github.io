"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _soft_min(a, b, k=50.0):
    """Smooth minimum using log-sum-exp trick."""
    m = np.minimum(a, b)
    return m - np.logaddexp(-k * (a - m), -k * (b - m)) / k


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


def _soft_min_penalty(z, k=30.0):
    """
    Dual-primal soft constraint: penalize violations smoothly using log-sum-exp.
    Returns a scalar penalty that is 0 when all constraints are satisfied.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return 1e6

    iu = np.triu_indices(N, k=1)

    # Pairwise: d_ij >= r_i + r_j  =>  violation = r_i + r_j - d_ij
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    viol_pairs = r_sum_pairs - d_pairs  # positive when violated

    # Soft-min of violations (smooth max)
    if np.max(viol_pairs) > 0:
        m = np.max(viol_pairs)
        soft_max_pairs = m + np.log(np.sum(np.exp(viol_pairs - m))) / k
    else:
        soft_max_pairs = 0.0

    # Boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_viol = np.concatenate([
        viol_pairs, left_viol, right_viol, bottom_viol, top_viol
    ])

    max_viol = np.max(all_viol)
    if max_viol > 0:
        m = max_viol
        soft_max_all = m + np.log(np.sum(np.exp(all_viol - m))) / k
    else:
        soft_max_all = 0.0

    return soft_max_all ** 2


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

        # Better grid layouts: use different row/col counts
        grid_configs = [
            (5, 5),  # 25 cells, use 21
            (4, 6),  # 24 cells, use 21
            (6, 4),  # 24 cells, use 21
            (3, 7),  # 21 cells exactly
            (7, 3),  # 21 cells exactly
            (2, 11), # 22 cells, use 21
            (11, 2), # 22 cells, use 21
        ]

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            for rows, cols in grid_configs:
                z = np.zeros(3 * N + 1)
                count = 0
                for row in range(rows):
                    for col in range(cols):
                        if count >= N:
                            break
                        cx = (col + 0.5) / cols * w0
                        cy = (row + 0.5) / rows * h0
                        r0 = min(w0 / cols, h0 / rows) * 0.45
                        z[3 * count] = cx
                        z[3 * count + 1] = cy
                        z[3 * count + 2] = r0
                        count += 1
                z[-1] = w0
                guesses.append(z)

        # Random starts with better distribution
        rng = np.random.default_rng(42)
        for trial in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            # Use Poisson-like random placement
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Sort guesses by initial objective (best first) to prioritize good starts
    def _initial_score(z):
        try:
            return _vectorized_objective(z, penalty_weight)
        except Exception:
            return 1e6

    guesses.sort(key=_initial_score)

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
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

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Dual-primal alternating optimization
        if best_result is not None and time.time() < deadline - 18:
            dual_result = _dual_primal_alternate(best_result, deadline - (time.time() - start_time) - 10)
            if dual_result is not None:
                valid, val = check_validity(dual_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = dual_result

        # 3. Iterative expansion with repositioning (multiple rounds)
        if best_result is not None and time.time() < deadline - 12:
            expanded = _iterative_expansion_polish(best_result, deadline - (time.time() - start_time) - 6)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 8:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final high-precision polish
        if best_result is not None and time.time() < deadline - 2:
            polished2 = _final_precision_polish(best_result, deadline - (time.time() - start_time))
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
    iu = np.triu_indices(n, k=1)

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

        # Pairwise repulsion (vectorized with np.add.at)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

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

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

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

        # Clamp positions (vectorized)
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

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
    iu = np.triu_indices(n, k=1)

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

        # Pairwise repulsion (vectorized with np.add.at)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

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

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

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

    # Try to grow radii with binary search (faster than incremental)
    lo_scale = 1.0
    hi_scale = 1.02
    for _ in range(25):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        r_trial = r * mid_scale
        if _is_valid_fast(x, y, r_trial, w, h):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

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


def _iterative_expansion_polish(z, time_budget):
    """
    Iterative expansion: repeatedly expand all circles, then reposition to make room,
    then expand again. Vectorized max-radius computation for speed with 8-direction search.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_outer_rounds = 25
    iu = np.triu_indices(n, k=1)

    # 8-direction unit vectors for local search
    directions = np.array([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]
    ]) / np.sqrt(2.0)

    for outer in range(max_outer_rounds):
        if time.time() > time_budget:
            break

        # Step 1: Vectorized - compute max radius for each circle
        # Wall limits
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        
        # Pairwise limits: for each i, min over j != i of (d_ij - r_j)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        # r_allowed[i,j] = d[i,j] - r[j], but we want min over j for each i
        r_allowed = d - r[None, :]  # shape (n, n)
        np.fill_diagonal(r_allowed, np.inf)
        r_pair_min = np.min(r_allowed, axis=1)  # min over j for each i
        
        r_max = np.minimum(r_wall, r_pair_min)
        r_max = np.maximum(r_max, 0.0)

        # Expand circles that can grow
        for i in range(n):
            if r_max[i] > r[i] + 1e-8:
                # Binary search for max valid radius (use fast check)
                lo = r[i]
                hi = r_max[i]
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid_fast(x, y, r_trial, w, h, tol=1e-9):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo

        # Step 2: 8-direction local search to find better positions
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Find the most constrained direction (closest to boundary or neighbor)
            min_gap = float('inf')
            best_dir = None
            for dx_dir, dy_dir in directions:
                # Check gap in this direction
                nx = x[i] + dx_dir * 0.01
                ny = y[i] + dy_dir * 0.01
                if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                    gap = 0.0
                else:
                    gap = float('inf')
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d2 = ddx * ddx + ddy * ddy
                        rs = r[i] + r[j]
                        if d2 < rs * rs:
                            gap = 0.0
                            break
                    if gap > float('inf') * 0.5:
                        gap = 0.01
                if gap < min_gap:
                    min_gap = gap
                    best_dir = (dx_dir, dy_dir)

            # Try moving in the best direction and opposite
            if best_dir is not None:
                for sign in [1.0, -1.0]:
                    for step in [0.02, 0.01, 0.005, 0.002]:
                        new_x = x[i] + sign * best_dir[0] * step
                        new_y = y[i] + sign * best_dir[1] * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            break
                    if improved:
                        break

        # Step 3: Force-directed relaxation to resolve overlaps
        x_new, y_new = _force_relax_positions(x, y, r, w, h, max_iter=200)
        if _is_valid_fast(x_new, y_new, r, w, h, tol=1e-9):
            x = x_new
            y = y_new
        else:
            # Try with smaller radii
            for shrink in [0.99, 0.98, 0.97]:
                r_shrunk = r * shrink
                if _is_valid_fast(x, y, r_shrunk, w, h, tol=1e-9):
                    r = r_shrunk
                    break
            else:
                break

    # Final: try uniform scaling up with binary search
    lo_scale = 1.0
    hi_scale = 1.05
    for _ in range(25):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        r_trial = r * mid_scale
        if _is_valid_fast(x, y, r_trial, w, h, tol=1e-9):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    if _is_valid(x, y, r, w, h, tol=1e-9):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _force_relax_positions(x, y, r, w, h, max_iter=200):
    """Pure position relaxation (radii fixed) to resolve overlaps - vectorized."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    n = N
    dt = 0.003
    iu = np.triu_indices(n, k=1)

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise repulsion - vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-10:
            break

        f_pairs = overlap * 10.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        # Vectorized force accumulation
        np.add.at(fx, iu[0], ux * f_pairs)
        np.add.at(fy, iu[0], uy * f_pairs)
        np.add.at(fx, iu[1], -ux * f_pairs)
        np.add.at(fy, iu[1], -uy * f_pairs)

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _final_precision_polish(z, time_budget):
    """
    High-precision final polish: alternating between vectorized position optimization
    and radius maximization with tight tolerances and more bisection iterations.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    iu = np.triu_indices(n, k=1)

    # Multiple alternating rounds of position polish + radius growth
    for round_num in range(15):
        if time.time() > time_budget:
            break

        # Position polish with small steps (vectorized)
        dt = 0.0002
        for _ in range(600):
            if time.time() > time_budget:
                break
            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall forces (gentle)
            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 15
            fx += right_viol * 15
            fy -= bottom_viol * 15
            fy += top_viol * 15

            # Pairwise (very gentle) - vectorized
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            if np.max(overlap) < 1e-14:
                break

            f_pairs = overlap * 6.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        # Try to grow each circle individually (vectorized max computation)
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        r_allowed = d - r[None, :]
        np.fill_diagonal(r_allowed, np.inf)
        r_pair_min = np.min(r_allowed, axis=1)
        r_max = np.maximum(np.minimum(r_wall, r_pair_min), 0.0)

        for i in range(n):
            if time.time() > time_budget:
                break
            if r_max[i] > r[i] + 1e-12:
                lo = r[i]
                hi = r_max[i]
                for _ in range(30):  # More bisection iterations for tighter precision
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid_fast(x, y, r_trial, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-13:
                    r[i] = lo

    # Final uniform scaling with high-precision bisection
    lo_scale = 1.0
    hi_scale = 1.01
    for _ in range(35):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        r_trial = r * mid_scale
        if _is_valid_fast(x, y, r_trial, w, h, tol=1e-10):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _dual_primal_alternate(z, time_budget):
    """
    Dual-primal alternating scheme:
    - Primal: optimize positions (x, y) given fixed radii and rectangle
    - Dual: optimize radii (r) and rectangle width (w) given fixed positions
    Uses smooth soft-min penalties for the nonsmooth constraint structure.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 12
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        # === PRIMAL PHASE: Optimize positions given fixed r, w ===
        # Use force-directed relaxation with adaptive step
        dt = 0.001
        for _ in range(300):
            if time.time() > time_budget:
                break
            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall forces
            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 25
            fx += right_viol * 25
            fy -= bottom_viol * 25
            fy += top_viol * 25

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

            if np.max(overlap) < 1e-12:
                break

            f_pairs = overlap * 12.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        # === DUAL PHASE: Optimize radii and w given fixed positions ===
        # Compute max radius for each circle
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        r_allowed = d - r[None, :]
        np.fill_diagonal(r_allowed, np.inf)
        r_pair_min = np.min(r_allowed, axis=1)
        r_max = np.maximum(np.minimum(r_wall, r_pair_min), 0.0)

        # Try to grow each circle
        for i in range(n):
            if time.time() > time_budget:
                break
            if r_max[i] > r[i] + 1e-10:
                lo = r[i]
                hi = r_max[i]
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid_fast(x, y, r_trial, w, h, tol=1e-9):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-11:
                    r[i] = lo

        # Try to adjust w to better fit the packing
        # Binary search for optimal w
        w_lo = max(0.3, w - 0.1)
        w_hi = min(1.7, w + 0.1)
        best_w = w
        best_val = -np.sum(r)
        for _ in range(15):
            if time.time() > time_budget:
                break
            w_mid = (w_lo + w_hi) / 2.0
            h_mid = 2.0 - w_mid
            # Check if current radii fit in new rectangle
            if _is_valid_fast(x, y, r, w_mid, h_mid, tol=1e-9):
                best_w = w_mid
                best_val = -np.sum(r)
                w_lo = w_mid
            else:
                w_hi = w_mid

        if best_w != w:
            w = best_w
            h = 2.0 - w

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
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid - vectorized."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Vectorized pairwise distance check
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
        return False
    return True


def _is_valid_fast(cx, cy, r, W, H, tol=1e-7):
    """Fast validity check with looser tolerance for inner loops."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
        return False
    return True
