"""Circle packing in perimeter-4 rectangle: SLSQP + Constraint Polish + LSE Refine + ADMM."""
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

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
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

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 25:
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

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Constraint-based SLSQP refinement (pure objective + explicit constraints)
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Smooth LSE epigraph refinement (smooth surrogate for non-convex dist constraints)
        if best_result is not None and time.time() < deadline - 10:
            lse = _lse_refine(best_result, deadline - (time.time() - start_time) - 5)
            if lse is not None:
                valid, val = check_validity(lse)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse

        # 5. Coordinate-descent polish: per-circle position + radius optimization
        if best_result is not None and time.time() < deadline - 8:
            cd = _coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 4)
            if cd is not None:
                valid, val = check_validity(cd)
                if valid and val > best_value:
                    best_value = val
                    best_result = cd

        # 6. Contact-graph optimization: exploit active constraint structure
        if best_result is not None and time.time() < deadline - 8:
            cg = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 5)
            if cg is not None:
                valid, val = check_validity(cg)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg

        # 7. Second LSE pass with tighter beta (exploit gains from CD/CG)
        if best_result is not None and time.time() < deadline - 6:
            lse2 = _lse_refine(best_result, deadline - (time.time() - start_time) - 3)
            if lse2 is not None:
                valid, val = check_validity(lse2)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse2

        # 8. Final coordinate-descent + expansion polish
        if best_result is not None and time.time() < deadline - 3:
            cd2 = _coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 1)
            if cd2 is not None:
                valid, val = check_validity(cd2)
                if valid and val > best_value:
                    best_value = val
                    best_result = cd2

        # 9. Final high-precision polish
        if best_result is not None and time.time() < deadline - 1:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
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
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
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

    dt = 0.002
    max_iter = int(min(5000, time_budget * 200))
    prev_best = float(np.sum(r))

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
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

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

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 20.0
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
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        # Only accept if valid
        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    # Try to grow radii uniformly with binary search
    for _ in range(50):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.03
        for _ in range(18):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-12:
            r = r * lo
        else:
            break

    # Per-circle greedy expansion with position search
    for _ in range(30):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            # First try to move circle i to create more space
            best_pos = None
            best_clear = -1.0
            for dx in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                    nx = x[i] + dx
                    ny = y[i] + dy
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    min_c = float('inf')
                    ok = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        gap = d - (r[i] + r[j])
                        if gap < -1e-9:
                            ok = False
                            break
                        if gap < min_c:
                            min_c = gap
                    if not ok:
                        continue
                    wc = min(nx - r[i], w - nx - r[i], ny - r[i], h - ny - r[i])
                    tc = min(min_c, wc)
                    if tc > best_clear:
                        best_clear = tc
                        best_pos = (nx, ny)

            if best_pos is not None and best_clear > 1e-8:
                x[i] = best_pos[0]
                y[i] = best_pos[1]

            # Now try to expand
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

            if r_max > r[i] + 1e-7:
                lo2 = r[i]
                hi2 = r_max
                for _ in range(18):
                    mid2 = (lo2 + hi2) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid2
                    if _is_valid(x, y, r_test, w, h):
                        lo2 = mid2
                    else:
                        hi2 = mid2
                if lo2 > r[i] + 1e-8:
                    r[i] = lo2
                    improved = True

        if not improved:
            break

    # Per-circle greedy expansion: try to grow each circle individually
    for _ in range(50):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Max radius for circle i given current positions
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

            if r_max > r[i] + 1e-7:
                lo2 = r[i]
                hi2 = r_max
                for _ in range(20):
                    mid2 = (lo2 + hi2) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid2
                    if _is_valid(x, y, r_test, w, h):
                        lo2 = mid2
                    else:
                        hi2 = mid2
                if lo2 > r[i] + 1e-8:
                    r[i] = lo2
                    improved = True

        if not improved:
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
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 8 directions
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    def try_expand_circle(i):
        """Try to expand circle i to max possible radius."""
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
            lo = r[i]
            hi = r_max
            for _ in range(25):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo
                return True
        return False

    def try_move_circle(i):
        """Try to move circle i in various directions to create space."""
        best_dir = None
        best_score = -float('inf')

        for dx_d, dy_d in dirs:
            new_x = x[i] + dx_d * 0.01
            new_y = y[i] + dy_d * 0.01
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue

            min_gap = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap

            wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
            score = min(min_gap, wall_dist)

            if score > best_score:
                best_score = score
                best_dir = (dx_d, dy_d)

        if best_dir is None:
            return False

        dx_d, dy_d = best_dir

        for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
            new_x = x[i] + dx_d * step
            new_y = y[i] + dy_d * step
            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                x[i] = new_x
                y[i] = new_y
                return True

        return False

    # Pass 1: Expand and move
    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            if try_expand_circle(i):
                improved = True

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

            if r_max - r[i] < 0.01:  # Limited expansion
                if try_move_circle(i):
                    improved = True

        if not improved:
            break

    # Pass 2: Second expansion after repositioning
    for rnd in range(20):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            if try_expand_circle(i):
                improved = True

        if not improved:
            break

    # Final uniform expansion with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
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


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using pure objective + explicit inequality constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        """Objective: minimize negative sum of radii (no penalty)."""
        r = zv[2::3][:n]
        return -np.sum(r)

    def make_circle_constraint(i):
        """Circle i must not overlap with any other circle: dist - (r[i]+r[j]) >= 0."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            dx = x[i] - x
            dy = y[i] - y
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            result = dist - (r[i] + r)
            result[i] = 1.0  # Self-constraint always satisfied
            return result
        return constraint

    def make_wall_constraint(axis, side):
        """Wall constraint: circle must stay inside rectangle."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            w = zv[-1]
            h = 2.0 - w

            if axis == 'x':
                pos = x
            else:
                pos = y

            if side == 'left' or side == 'bottom':
                return pos - r
            else:
                if axis == 'x':
                    return w - pos - r
                else:
                    return h - pos - r
        return constraint

    # Build constraints
    constraints = []

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_circle_constraint(i)})

    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'left')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'right')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'bottom')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'top')})

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 300))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        # Try to repair if slightly invalid
        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _lse_refine(z, time_budget):
    """
    Smooth surrogate refinement: minimize -sum(r) + smooth log-sum-exp penalty
    on overlap and wall violations. Iterative LSE with decreasing beta for
    progressive constraint tightening.
    """
    z = z.copy().astype(float)
    n = N
    best_z = z.copy()
    best_val = float(np.sum(z[2::3][:n]))

    # Iterative LSE: start with moderate beta, tighten over iterations
    beta_schedule = [100.0, 200.0, 400.0, 800.0, 1500.0]
    weight_schedule = [30.0, 50.0, 80.0, 100.0, 120.0]

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    for it, (beta, weight) in enumerate(zip(beta_schedule, weight_schedule)):
        if time.time() > time_budget - 2:
            break

        def lse_obj(zv, beta=beta, weight=weight):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            w = zv[-1]
            h = 2.0 - w

            if w < 0.2 or h < 0.2 or np.any(r < 0):
                return 1e6

            # Pairwise overlap: r_i + r_j - dist
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            iu = np.triu_indices(n, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            viol = r_sum_pairs - d_pairs  # > 0 when overlapping

            # Wall violations (all four)
            wall_viol = np.concatenate([
                np.maximum(r - x, 0.0),
                np.maximum(x + r - w, 0.0),
                np.maximum(r - y, 0.0),
                np.maximum(y + r - h, 0.0),
            ])

            all_viol = np.concatenate([viol, wall_viol])
            m = np.max(all_viol)
            lse = m + np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta

            return -np.sum(r) + weight * lse

        remaining_budget = time_budget - (time.time() - (time_budget - time_budget))  # placeholder
        remaining = time_budget - time.time()
        if remaining < 3:
            break

        maxiter = int(min(800, remaining * 100))
        try:
            res = minimize(lse_obj, best_z, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 120})

            z_opt = res.x
            x = z_opt[0::3][:n]
            y = z_opt[1::3][:n]
            r = z_opt[2::3][:n]
            w = z_opt[-1]
            h = 2.0 - w

            if not _is_valid(x, y, r, w, h):
                for factor in [0.9999, 0.999, 0.998, 0.997, 0.995, 0.99, 0.98]:
                    r_trial = r * factor
                    if _is_valid(x, y, r_trial, w, h):
                        r = r_trial
                        break
                else:
                    continue

            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                z_opt[2::3][:n] = r
                best_z = z_opt.copy()

        except Exception:
            continue

    if best_val > float(np.sum(z[2::3][:n])):
        return best_z
    return None


def _admm_refine(z, time_budget):
    """
    ADMM refinement: Alternating Direction Method of Multipliers.
    Decouples circle-circle non-overlap constraints from wall constraints and radius maximization.
    Uses augmented Lagrangian with dual updates for better convergence.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # ADMM parameters
    rho = 10.0  # Penalty parameter
    max_iter = int(min(300, time_budget * 50))

    # Dual variables for pairwise constraints
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    dual_pair = np.zeros(n_pairs)

    # Dual variables for wall constraints
    dual_left = np.zeros(n)
    dual_right = np.zeros(n)
    dual_bottom = np.zeros(n)
    dual_top = np.zeros(n)

    def solve_x_subproblem(x_curr, y_curr, r_curr, w_curr, h_curr, dual_p, dual_l, dual_r, dual_b, dual_t, rho_curr):
        """Solve x-update: minimize -sum(r) + augmented Lagrangian for x."""
        x_new = x_curr.copy()
        for _ in range(50):  # Inner iterations for x
            if time.time() > time_budget:
                break

            grad_x = np.zeros(n)

            # Wall augmented Lagrangian
            for i in range(n):
                # Left wall: x_i - r_i >= 0
                viol_l = max(0, r_curr[i] - x_new[i])
                grad_x[i] += rho_curr * viol_l - dual_l[i]

                # Right wall: w - x_i - r_i >= 0
                viol_r = max(0, x_new[i] + r_curr[i] - w_curr)
                grad_x[i] -= rho_curr * viol_r + dual_r[i]

            # Pairwise augmented Lagrangian
            for idx in range(n_pairs):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                dx = x_new[i_idx] - x_new[j_idx]
                dy = y_curr[i_idx] - y_curr[j_idx]
                d_sq = dx * dx + dy * dy
                d = np.sqrt(max(d_sq, 1e-12))
                r_sum = r_curr[i_idx] + r_curr[j_idx]
                viol = max(0, r_sum - d)

                if viol > 0:
                    # Gradient of augmented Lagrangian w.r.t. x
                    ux = dx / d
                    uy = dy / d
                    force = rho_curr * viol - dual_p[idx]
                    grad_x[i_idx] += ux * force
                    grad_x[j_idx] -= ux * force

            # Gradient descent step
            step = 0.01 / (1 + np.max(np.abs(grad_x)) * 10)
            x_new -= step * grad_x

            # Clamp to walls
            for i in range(n):
                x_new[i] = np.clip(x_new[i], r_curr[i], w_curr - r_curr[i])

        return x_new

    def solve_y_subproblem(x_curr, y_curr, r_curr, w_curr, h_curr, dual_p, dual_l, dual_r, dual_b, dual_t, rho_curr):
        """Solve y-update: minimize -sum(r) + augmented Lagrangian for y."""
        y_new = y_curr.copy()
        for _ in range(50):  # Inner iterations for y
            if time.time() > time_budget:
                break

            grad_y = np.zeros(n)

            # Wall augmented Lagrangian
            for i in range(n):
                # Bottom wall: y_i - r_i >= 0
                viol_b = max(0, r_curr[i] - y_new[i])
                grad_y[i] += rho_curr * viol_b - dual_b[i]

                # Top wall: h - y_i - r_i >= 0
                viol_t = max(0, y_new[i] + r_curr[i] - h_curr)
                grad_y[i] -= rho_curr * viol_t + dual_t[i]

            # Pairwise augmented Lagrangian
            for idx in range(n_pairs):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                dx = x_curr[i_idx] - x_curr[j_idx]
                dy = y_new[i_idx] - y_new[j_idx]
                d_sq = dx * dx + dy * dy
                d = np.sqrt(max(d_sq, 1e-12))
                r_sum = r_curr[i_idx] + r_curr[j_idx]
                viol = max(0, r_sum - d)

                if viol > 0:
                    ux = dx / d
                    uy = dy / d
                    force = rho_curr * viol - dual_p[idx]
                    grad_y[i_idx] += uy * force
                    grad_y[j_idx] -= uy * force

            # Gradient descent step
            step = 0.01 / (1 + np.max(np.abs(grad_y)) * 10)
            y_new -= step * grad_y

            # Clamp to walls
            for i in range(n):
                y_new[i] = np.clip(y_new[i], r_curr[i], h_curr - r_curr[i])

        return y_new

    def solve_r_subproblem(x_curr, y_curr, r_curr, w_curr, h_curr, dual_p, dual_l, dual_r, dual_b, dual_t, rho_curr):
        """Solve r-update: maximize sum(r) subject to constraints."""
        r_new = r_curr.copy()
        for _ in range(50):  # Inner iterations for r
            if time.time() > time_budget:
                break

            grad_r = np.zeros(n)

            # Wall augmented Lagrangian
            for i in range(n):
                viol_l = max(0, r_new[i] - x_curr[i])
                grad_r[i] -= rho_curr * viol_l + dual_l[i]

                viol_r = max(0, x_curr[i] + r_new[i] - w_curr)
                grad_r[i] -= rho_curr * viol_r + dual_r[i]

                viol_b = max(0, r_new[i] - y_curr[i])
                grad_r[i] -= rho_curr * viol_b + dual_b[i]

                viol_t = max(0, y_curr[i] + r_new[i] - h_curr)
                grad_r[i] -= rho_curr * viol_t + dual_t[i]

            # Pairwise augmented Lagrangian
            for idx in range(n_pairs):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                dx = x_curr[i_idx] - x_curr[j_idx]
                dy = y_curr[i_idx] - y_curr[j_idx]
                d_sq = dx * dx + dy * dy
                d = np.sqrt(max(d_sq, 1e-12))
                r_sum = r_new[i_idx] + r_new[j_idx]
                viol = max(0, r_sum - d)

                if viol > 0:
                    force = rho_curr * viol - dual_p[idx]
                    grad_r[i_idx] -= force
                    grad_r[j_idx] -= force

            # Gradient ascent step (maximize sum(r))
            step = 0.01 / (1 + np.max(np.abs(grad_r)) * 10)
            r_new += step * grad_r

            # Clamp to positive
            r_new = np.maximum(r_new, 0.01)

        return r_new

    # ADMM iterations
    for it in range(max_iter):
        if time.time() > time_budget:
            break

        # X update
        x_new = solve_x_subproblem(x, y, r, w, h, dual_pair, dual_left, dual_right, dual_bottom, dual_top, rho)

        # Y update
        y_new = solve_y_subproblem(x_new, y, r, w, h, dual_pair, dual_left, dual_right, dual_bottom, dual_top, rho)

        # R update
        r_new = solve_r_subproblem(x_new, y_new, r, w, h, dual_pair, dual_left, dual_right, dual_bottom, dual_top, rho)

        # Dual updates
        for idx in range(n_pairs):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            dx = x_new[i_idx] - x_new[j_idx]
            dy = y_new[i_idx] - y_new[j_idx]
            d_sq = dx * dx + dy * dy
            d = np.sqrt(max(d_sq, 1e-12))
            r_sum = r_new[i_idx] + r_new[j_idx]
            viol = max(0, r_sum - d)
            dual_pair[idx] += rho * viol

        for i in range(n):
            dual_left[i] += rho * max(0, r_new[i] - x_new[i])
            dual_right[i] += rho * max(0, x_new[i] + r_new[i] - w)
            dual_bottom[i] += rho * max(0, r_new[i] - y_new[i])
            dual_top[i] += rho * max(0, y_new[i] + r_new[i] - h)

        # Update variables
        x = x_new
        y = y_new
        r = r_new

        # Adaptive rho
        if it % 50 == 49:
            rho *= 1.1

    # Check validity and repair if needed
    if not _is_valid(x, y, r, w, h):
        for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
            r_trial = r * factor
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
                break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _contact_graph_optimize(z, time_budget):
    """
    Contact-graph based local optimization: identify the set of active constraints
    (circles touching each other or walls), then solve a smaller optimization problem
    where only the active constraints are enforced as equalities. This is a
    differentiable approach that avoids the non-convexity of the full distance
    constraints by working on the contact structure.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def get_active_constraints(x_curr, y_curr, r_curr, tol=1e-6):
        """Identify active constraints: pairwise contacts and wall contacts."""
        active_pairs = []
        active_walls = []  # list of (circle_idx, wall_type)

        for i in range(n):
            for j in range(i + 1, n):
                dx = x_curr[i] - x_curr[j]
                dy = y_curr[i] - y_curr[j]
                d = np.sqrt(dx * dx + dy * dy)
                if abs(d - (r_curr[i] + r_curr[j])) < tol:
                    active_pairs.append((i, j))

            if abs(x_curr[i] - r_curr[i]) < tol:
                active_walls.append((i, 'left'))
            if abs(x_curr[i] + r_curr[i] - w) < tol:
                active_walls.append((i, 'right'))
            if abs(y_curr[i] - r_curr[i]) < tol:
                active_walls.append((i, 'bottom'))
            if abs(y_curr[i] + r_curr[i] - h) < tol:
                active_walls.append((i, 'top'))

        return active_pairs, active_walls

    def objective_cg(zv):
        """Objective: minimize -sum(r) with smooth penalty for inactive constraint violations."""
        xv = zv[0::3][:n]
        yv = zv[1::3][:n]
        rv = zv[2::3][:n]
        wv = zv[-1]
        hv = 2.0 - wv

        if wv < 0.2 or hv < 0.2 or np.any(rv < 0):
            return 1e6

        obj = -np.sum(rv)

        # Smooth penalty for pairwise overlaps
        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rv[:, None] + rv[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        obj += 100.0 * np.sum(overlap * overlap)

        # Wall penalties
        obj += 100.0 * (
            np.sum(np.maximum(rv - xv, 0.0) ** 2) +
            np.sum(np.maximum(xv + rv - wv, 0.0) ** 2) +
            np.sum(np.maximum(rv - yv, 0.0) ** 2) +
            np.sum(np.maximum(yv + rv - hv, 0.0) ** 2)
        )

        return obj

    def constraints_cg(zv):
        """Active constraints as smooth equalities (squared to keep inequality form)."""
        xv = zv[0::3][:n]
        yv = zv[1::3][:n]
        rv = zv[2::3][:n]
        wv = zv[-1]
        hv = 2.0 - wv

        active_pairs, active_walls = get_active_constraints(xv, yv, rv)

        # Pairwise: dist >= r_i + r_j  =>  dist^2 - (r_i+r_j)^2 >= 0
        # Use smooth version: (dist - (r_i+r_j)) >= 0
        pair_constraints = []
        for (i, j) in active_pairs:
            dx = xv[i] - xv[j]
            dy = yv[i] - yv[j]
            d = np.sqrt(max(dx * dx + dy * dy, 1e-12))
            pair_constraints.append(d - (rv[i] + rv[j]))

        # Wall constraints
        wall_constraints = []
        for (i, wall) in active_walls:
            if wall == 'left':
                wall_constraints.append(xv[i] - rv[i])
            elif wall == 'right':
                wall_constraints.append(wv - xv[i] - rv[i])
            elif wall == 'bottom':
                wall_constraints.append(yv[i] - rv[i])
            else:
                wall_constraints.append(hv - yv[i] - rv[i])

        return np.array(pair_constraints + wall_constraints) if (pair_constraints or wall_constraints) else np.array([1.0])

    # Iterative contact-graph refinement
    best_val = float(np.sum(r))
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for outer_iter in range(20):
        if time.time() > time_budget:
            break

        # Get current active constraints
        active_pairs, active_walls = get_active_constraints(x, y, r, tol=1e-5)

        if len(active_pairs) + len(active_walls) < 3:
            # Not enough contacts to form a rigid structure; use full constraint set
            break

        # Build SLSQP problem with active constraints
        def obj_func(zv):
            return objective_cg(zv)

        cons_list = []
        for (i, j) in active_pairs:
            def make_pair_con(i=i, j=j):
                def con(zv):
                    xv = zv[0::3][:n]
                    yv = zv[1::3][:n]
                    rv = zv[2::3][:n]
                    dx = xv[i] - xv[j]
                    dy = yv[i] - yv[j]
                    d = np.sqrt(max(dx * dx + dy * dy, 1e-12))
                    return d - (rv[i] + rv[j])
                return con
            cons_list.append({'type': 'ineq', 'fun': make_pair_con()})

        for (i, wall) in active_walls:
            def make_wall_con(i=i, wall=wall):
                def con(zv):
                    xv = zv[0::3][:n]
                    yv = zv[1::3][:n]
                    rv = zv[2::3][:n]
                    wv = zv[-1]
                    hv = 2.0 - wv
                    if wall == 'left':
                        return xv[i] - rv[i]
                    elif wall == 'right':
                        return wv - xv[i] - rv[i]
                    elif wall == 'bottom':
                        return yv[i] - rv[i]
                    else:
                        return hv - yv[i] - rv[i]
                return con
            cons_list.append({'type': 'ineq', 'fun': make_wall_con()})

        bounds = []
        for i in range(n):
            bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
        bounds.append((0.3, 1.7))

        maxiter = int(min(500, time_budget * 100))
        try:
            res = minimize(obj_func, z, method='SLSQP',
                           bounds=bounds,
                           constraints=cons_list if cons_list else [],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})

            z_opt = res.x
            x_new = z_opt[0::3][:n].copy()
            y_new = z_opt[1::3][:n].copy()
            r_new = z_opt[2::3][:n].copy()
            w_new = z_opt[-1]
            h_new = 2.0 - w_new

            if _is_valid(x_new, y_new, r_new, w_new, h_new):
                val = float(np.sum(r_new))
                if val > best_val:
                    best_val = val
                    best_x, best_y, best_r = x_new, y_new, r_new
                    x, y, r, w, h = x_new, y_new, r_new, w_new, h_new
            else:
                # Try slight shrink
                for factor in [0.999, 0.998, 0.997]:
                    r_trial = r_new * factor
                    if _is_valid(x_new, y_new, r_trial, w_new, h_new):
                        val = float(np.sum(r_trial))
                        if val > best_val:
                            best_val = val
                            best_x, best_y, best_r = x_new, y_new, r_trial
                            x, y, r, w, h = x_new, y_new, r_trial, w_new, h_new
                        break

        except Exception:
            continue

    if best_val > float(np.sum(z[2::3][:n])):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = w
        return z_new
    return None


def _coordinate_descent_polish(z, time_budget):
    """
    Coordinate-descent polish: alternate per-circle position optimization and
    radius expansion. For each circle, try to move it to a position that
    maximizes the minimum clearance, then expand its radius to the maximum
    possible given neighbors.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_val = float(np.sum(r))

    for outer_iter in range(30):
        if time.time() > time_budget:
            break

        improved = False

        for i in range(n):
            if time.time() > time_budget:
                break

            # Try to find a better position for circle i
            best_pos = None
            best_clearance = -1.0

            # Candidate positions: current + small perturbations in 8 directions
            for dx in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
                for dy in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
                    new_x = x[i] + dx
                    new_y = y[i] + dy

                    if new_x < r[i] or new_x > w - r[i]:
                        continue
                    if new_y < r[i] or new_y > h - r[i]:
                        continue

                    # Check validity and compute min clearance
                    min_clear = float('inf')
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = new_x - x[j]
                        ddy = new_y - y[j]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        gap = d - (r[i] + r[j])
                        if gap < -1e-9:
                            valid = False
                            break
                        if gap < min_clear:
                            min_clear = gap

                    if not valid:
                        continue

                    wall_clear = min(new_x - r[i], w - new_x - r[i],
                                     new_y - r[i], h - new_y - r[i])
                    total_clear = min(min_clear, wall_clear)

                    if total_clear > best_clearance:
                        best_clearance = total_clear
                        best_pos = (new_x, new_y)

            if best_pos is not None and best_clearance > 1e-8:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                improved = True

        # After repositioning, try to expand each circle
        for i in range(n):
            if time.time() > time_budget:
                break

            # Max radius for circle i
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

            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Final uniform expansion
    for _ in range(50):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.02
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h) and float(np.sum(r)) > best_val:
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


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
