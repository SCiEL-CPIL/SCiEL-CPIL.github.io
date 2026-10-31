"""Circle packing in perimeter-4 rectangle: Vectorized SLSQP + Multi-Phase Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
IU = np.triu_indices(N, k=1)  # precompute upper-triangle indices


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
    dx = x[IU[0]] - x[IU[1]]
    dy = y[IU[0]] - y[IU[1]]
    d_sq = dx * dx + dy * dy
    r_sum = r[IU[0]] + r[IU[1]]

    d_pairs = np.sqrt(np.maximum(d_sq, 0.0))
    overlap = np.maximum(r_sum - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

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
    best_z = None

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

    def force_relax(z0, max_iter=500):
        """Force-directed relaxation to resolve overlaps."""
        z = z0.copy()
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        dt = 0.01
        for _ in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)

            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 100
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 100
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 100
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 100

            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        overlap = r_sum - d
                        if d > 1e-6:
                            fx[i] += (dx / d) * overlap * 10
                            fy[i] += (dy / d) * overlap * 10
                            fx[j] -= (dx / d) * overlap * 10
                            fy[j] -= (dy / d) * overlap * 10
                        else:
                            fx[i] += np.random.uniform(-1, 1) * overlap * 10
                            fy[i] += np.random.uniform(-1, 1) * overlap * 10
                            fx[j] -= np.random.uniform(-1, 1) * overlap * 10
                            fy[j] -= np.random.uniform(-1, 1) * overlap * 10

            x += dt * fx
            y += dt * fy

            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

            # Convergence check
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < r[i] + r[j]:
                        max_overlap = max(max_overlap, r[i] + r[j] - d)
            if max_overlap < 1e-6:
                break

        z[0::3][:N] = x
        z[1::3][:N] = y
        return z

    def expand_radii(z, time_budget):
        """Iteratively expand radii to maximum possible via bisection."""
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        def get_max_radius(i):
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

        max_rounds = 20
        for _ in range(max_rounds):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(N):
                r_max = get_max_radius(i)
                if r_max > r[i] + 1e-6:
                    lo = r[i]
                    hi = r_max
                    for _ in range(15):
                        mid = (lo + hi) / 2.0
                        r[i] = mid
                        if check_validity(z)[0]:
                            lo = mid
                        else:
                            hi = mid
                    r[i] = lo
                    if r[i] > r[i] * (1 + 1e-6):
                        improved = True
            if not improved:
                break

        z[2::3][:N] = r
        return z

    def local_search_polish(z, time_budget):
        """8-direction local search: nudge positions to create space, then expand."""
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        step_sizes = [0.005, 0.01, 0.02, 0.002]

        best_local = z.copy()
        best_local_val = np.sum(r)

        for _ in range(50):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(N):
                for ang in angles:
                    for step in step_sizes:
                        nx = x[i] + step * np.cos(ang)
                        ny = y[i] + step * np.sin(ang)
                        # Check bounds
                        if nx < r[i] or nx > w - r[i]:
                            continue
                        if ny < r[i] or ny > h - r[i]:
                            continue
                        # Check no overlap introduced
                        ok = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = nx - x[j]
                            dy = ny - y[j]
                            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                                ok = False
                                break
                        if ok:
                            x[i] = nx
                            y[i] = ny
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break

        z[0::3][:N] = x
        z[1::3][:N] = y
        return z

    def constraint_refine(z, time_budget):
        """SLSQP with explicit constraint functions (not penalty)."""
        x0 = z[0::3][:N].copy()
        y0 = z[1::3][:N].copy()
        r0 = z[2::3][:N].copy()
        w0 = z[-1]

        def obj(zc):
            return -np.sum(zc[2::3][:N])

        constraints = []
        # Pairwise distance constraints
        for i in range(N):
            for j in range(i + 1, N):
                def c(zc, i=i, j=j):
                    dx = zc[3 * i] - zc[3 * j]
                    dy = zc[3 * i + 1] - zc[3 * j + 1]
                    ri = zc[3 * i + 2]
                    rj = zc[3 * j + 2]
                    return dx * dx + dy * dy - (ri + rj) ** 2
                constraints.append({'type': 'ineq', 'fun': c})

        # Boundary constraints
        for i in range(N):
            def c_left(zc, i=i):
                return zc[3 * i] - zc[3 * i + 2]
            def c_right(zc, i=i):
                return zc[-1] - zc[3 * i] - zc[3 * i + 2]
            def c_bottom(zc, i=i):
                return zc[3 * i + 1] - zc[3 * i + 2]
            def c_top(zc, i=i):
                return (2.0 - zc[-1]) - zc[3 * i + 1] - zc[3 * i + 2]
            constraints.append({'type': 'ineq', 'fun': c_left})
            constraints.append({'type': 'ineq', 'fun': c_right})
            constraints.append({'type': 'ineq', 'fun': c_bottom})
            constraints.append({'type': 'ineq', 'fun': c_top})

        zc0 = z.copy()
        bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

        try:
            res = minimize(obj, zc0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 100})
            if res.fun < obj(zc0) + 1e-8:
                valid, val = check_validity(res.x)
                if valid:
                    return res.x
        except Exception:
            pass
        return None

    def lse_refine(z, time_budget):
        """Log-sum-exp smooth surrogate for non-convex distance constraints."""
        x0 = z[0::3][:N].copy()
        y0 = z[1::3][:N].copy()
        r0 = z[2::3][:N].copy()
        w0 = z[-1]

        def lse_obj(zc):
            x_ = zc[0::3][:N]
            y_ = zc[1::3][:N]
            r_ = zc[2::3][:N]
            w_ = zc[-1]
            h_ = 2.0 - w_

            if w_ < 0.2 or h_ < 0.2:
                return 1e6

            base = -np.sum(r_)

            # LSE for pairwise constraints: smooth max of (r_i + r_j - d_ij, 0)
            dx = x_[IU[0]] - x_[IU[1]]
            dy = y_[IU[0]] - y_[IU[1]]
            d_sq = dx * dx + dy * dy
            r_sum = r_[IU[0]] + r_[IU[1]]
            d = np.sqrt(np.maximum(d_sq, 1e-12))
            violations = r_sum - d
            # LSE approximation with temperature
            temp = 0.01
            max_v = np.max(violations)
            if max_v > 0:
                lse_val = temp * np.log(np.sum(np.exp(violations / temp)))
            else:
                lse_val = 0.0
            base += 500.0 * lse_val

            # Boundary LSE
            left_v = np.maximum(r_ - x_, 0.0)
            right_v = np.maximum(x_ + r_ - w_, 0.0)
            bottom_v = np.maximum(r_ - y_, 0.0)
            top_v = np.maximum(y_ + r_ - h_, 0.0)
            all_v = np.concatenate([left_v, right_v, bottom_v, top_v])
            max_bv = np.max(all_v)
            if max_bv > 0:
                lse_b = temp * np.log(np.sum(np.exp(all_v / temp)))
            else:
                lse_b = 0.0
            base += 500.0 * lse_b

            return base

        bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
        try:
            res = minimize(lse_obj, z.copy(), method='SLSQP', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 80})
            valid, val = check_validity(res.x)
            if valid:
                return res.x
        except Exception:
            pass
        return None

    def high_precision_expand(z, time_budget):
        """Final high-precision radius expansion with bisection at tol=1e-10."""
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        def get_max_radius(i):
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

        for _ in range(10):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(N):
                r_max = get_max_radius(i)
                if r_max > r[i] + 1e-10:
                    lo = r[i]
                    hi = r_max
                    for _ in range(30):
                        mid = (lo + hi) / 2.0
                        r[i] = mid
                        if check_validity(z)[0]:
                            lo = mid
                        else:
                            hi = mid
                    r[i] = lo
                    if r[i] > r[i] * (1 + 1e-10):
                        improved = True
            if not improved:
                break

        z[2::3][:N] = r
        return z

    # Generate diverse initial guesses
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
    for _ in range(12):
        w0 = rng.uniform(0.55, 1.45)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.04, 0.13)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    # Phase 1: SLSQP with vectorized objective from multiple starts
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Force-directed relaxation
        z_rel = force_relax(z0, max_iter=400)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 40)))
        try:
            res = minimize(_vectorized_objective, z_rel, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_z = z_opt.copy()
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_z = z_repair.copy()

        except Exception:
            continue

    # Phase 2: Multi-stage polish on best result
    if best_z is not None:
        # 2a. Local search + expansion
        if time.time() < deadline - 25:
            polished = local_search_polish(best_z, deadline - 15)
            polished = expand_radii(polished, deadline - 12)
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_z = polished

        # 2b. Constraint-based SLSQP refinement
        if time.time() < deadline - 18:
            refined = constraint_refine(best_z, deadline - 10)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_z = refined

        # 2c. LSE epigraph refinement
        if time.time() < deadline - 12:
            lse = lse_refine(best_z, deadline - 6)
            if lse is not None:
                valid, val = check_validity(lse)
                if valid and val > best_value:
                    best_value = val
                    best_z = lse

        # 2d. Second constraint refinement
        if time.time() < deadline - 8:
            refined2 = constraint_refine(best_z, deadline - 4)
            if refined2 is not None:
                valid, val = check_validity(refined2)
                if valid and val > best_value:
                    best_value = val
                    best_z = refined2

        # 2e. Local search + high-precision expansion
        if time.time() < deadline - 5:
            polished2 = local_search_polish(best_z, deadline - 3)
            polished2 = high_precision_expand(polished2, deadline - 1)
            valid, val = check_validity(polished2)
            if valid and val > best_value:
                best_value = val
                best_z = polished2

    if best_z is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety: shrink if invalid
    z = best_z.copy()
    for _ in range(10):
        if check_validity(z)[0]:
            break
        r_idx = np.arange(2, len(z) - 1, 3)
        z[r_idx] *= 0.99

    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    return np.column_stack((x, y, r))
