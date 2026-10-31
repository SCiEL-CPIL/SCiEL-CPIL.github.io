"""Circle packing in perimeter-4 rectangle: Smooth epigraph optimization with SLSQP polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    best_value = 0.0
    best_result = None

    iu = np.triu_indices(N, k=1)

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def is_valid_fast(x, y, r, w, h, tol=1e-8):
        if w <= 0.05 or h <= 0.05:
            return False
        if np.any(r <= 0):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False
        return True

    def smooth_objective(z):
        """
        z = [x_0, y_0, r_0, ..., x_20, y_20, r_20, w]
        Minimize -sum(r) + soft penalty for constraint violations.
        Uses smooth soft-min approximation for overlap constraints.
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.15 or h < 0.15:
            return 1e6

        obj = -np.sum(r)

        # Pairwise: smooth penalty for d < r_i + r_j
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        # violation = max(r_sum - d, 0), use smooth version
        d = np.sqrt(np.maximum(d2, 1e-12))
        viol = r_sum - d
        # soft penalty: beta * (log(1+exp(viol/beta)) - log(2)) approximates max(viol,0)
        beta = 0.01
        # For stability, use: if viol > 0: viol + beta*log(1-exp(-viol/beta)) ~ viol
        # Simpler: quadratic on positive part with smooth transition
        smooth_viol = np.where(viol > 0, viol + beta * np.log1p(-np.exp(-np.maximum(viol, 0) / beta)), 0)
        # Actually let's just use a smooth quadratic
        obj += 500.0 * np.sum(np.maximum(viol, 0.0) ** 2)
        # Add a small linear term to encourage separation
        obj += 50.0 * np.sum(np.maximum(viol, 0.0))

        # Boundary violations
        left_v = np.maximum(r - x, 0.0)
        right_v = np.maximum(x + r - w, 0.0)
        bottom_v = np.maximum(r - y, 0.0)
        top_v = np.maximum(y + r - h, 0.0)
        obj += 500.0 * (np.sum(left_v**2) + np.sum(right_v**2) + np.sum(bottom_v**2) + np.sum(top_v**2))
        obj += 50.0 * (np.sum(left_v) + np.sum(right_v) + np.sum(bottom_v) + np.sum(top_v))

        # Keep aspect ratio reasonable
        if w < 0.5:
            obj += 100.0 * (0.5 - w) ** 2
        elif w > 1.5:
            obj += 100.0 * (w - 1.5) ** 2

        return obj

    def smooth_gradient(z):
        """Numerical gradient (fast enough for 64-dim problem)."""
        eps = 1e-7
        g = np.zeros_like(z)
        f0 = smooth_objective(z)
        for i in range(len(z)):
            z_old = z[i]
            z[i] = z_old + eps
            f1 = smooth_objective(z)
            z[i] = z_old
            g[i] = (f1 - f0) / eps
        return g

    def force_relax(x, y, r, w, h, max_iter=2000, tol=1e-11):
        """Force-directed relaxation with adaptive step size."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)

        dt = 0.01
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            fx = np.zeros(N)
            fy = np.zeros(N)

            # Wall forces
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 100
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 100
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 100
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 100

            # Pairwise repulsion
            max_overlap = 0.0
            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-12:
                    d = 1e-12
                    dx = 1e-6
                    dy = 0
                r_sum = r[i] + r[j]
                overlap = r_sum - d
                if overlap > 0:
                    max_overlap = max(max_overlap, overlap)
                    f = overlap * 50.0
                    fx[i] += (dx / d) * f
                    fy[i] += (dy / d) * f
                    fx[j] -= (dx / d) * f
                    fy[j] -= (dy / d) * f

            if max_overlap < tol:
                break

            # Adaptive step
            max_f = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-10)
            dt = 0.5 / max_f if max_f > 0.01 else 0.005
            dt = min(dt, 0.02)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        return x, y, r

    def expand_radii(x, y, r, w, h, time_budget=5.0):
        """Greedily expand each radius to its maximum valid value."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()

        for _ in range(30):
            if time.time() - start > time_budget:
                break
            improved = False
            order = np.argsort(r)
            for i in order:
                if time.time() - start > time_budget:
                    break
                # Max from walls
                r_max = min(x[i], w - x[i], y[i], h - y[i])
                # Max from other circles
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
                if r_max > r[i] + 1e-9:
                    r[i] = r_max
                    improved = True
            if not improved:
                break

        return x, y, r

    def local_search(x, y, r, w, h, time_budget=10.0):
        """Local search: try moving circles to create more space."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()

        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        steps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]

        for rnd in range(30):
            if time.time() - start > time_budget:
                break
            improved = False
            for i in range(N):
                if time.time() - start > time_budget:
                    break
                best_move = None
                best_gain = 1e-7

                for angle in angles:
                    for step in steps:
                        nx = x[i] + step * np.cos(angle)
                        ny = y[i] + step * np.sin(angle)
                        if nx < r[i] + 1e-9 or nx > w - r[i] - 1e-9:
                            continue
                        if ny < r[i] + 1e-9 or ny > h - r[i] - 1e-9:
                            continue

                        # Check no overlap
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = nx - x[j]
                            dy = ny - y[j]
                            d2 = dx * dx + dy * dy
                            rs = r[i] + r[j]
                            if d2 < rs * rs - 1e-9:
                                valid = False
                                break
                        if not valid:
                            continue

                        # Compute new max radius
                        r_wall = min(nx, w - nx, ny, h - ny)
                        r_max_new = r_wall
                        for j in range(N):
                            if i == j:
                                continue
                            dx = nx - x[j]
                            dy = ny - y[j]
                            d = np.sqrt(dx * dx + dy * dy)
                            r_allowed = d - r[j]
                            if r_allowed < r_max_new:
                                r_max_new = r_allowed
                        r_max_new = max(0.0, r_max_new)

                        gain = r_max_new - r[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (nx, ny, r_max_new)

                if best_move is not None:
                    x[i] = best_move[0]
                    y[i] = best_move[1]
                    r[i] = best_move[2]
                    improved = True

            if not improved:
                break

        return x, y, r

    def slsqp_polish(z, time_budget=15.0):
        """SLSQP with explicit inequality constraints for final polish."""
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        def obj(zv):
            return -np.sum(zv[2::3][:N])

        constraints = []
        bounds = []

        for i in range(N):
            bounds.extend([
                (r[i] * 0.3, min(x[i], w - x[i], y[i], h - y[i]) * 2.0 + 0.01),
                (r[i] * 0.3, min(y[i], h - y[i]) * 2.0 + 0.01),
                (0.005, 1.0)
            ])
        bounds.append((max(0.3, w - 0.15), min(1.7, w + 0.15)))

        # Boundary constraints: g >= 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

        # Overlap constraints
        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })

        maxiter = int(min(800, time_budget * 60))
        try:
            res = minimize(obj, z, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            if res.success or (res.fun < -best_value + 1e-6):
                zo = res.x
                xo = zo[0::3][:N]
                yo = zo[1::3][:N]
                ro = zo[2::3][:N]
                wo = zo[-1]
                ho = 2.0 - wo
                if is_valid_fast(xo, yo, ro, wo, ho):
                    return zo
        except Exception:
            pass
        return None

    def lbfgsb_optimize(z, time_budget=10.0):
        """L-BFGS-B on smooth penalized objective."""
        try:
            maxiter = int(min(500, time_budget * 50))
            res = minimize(smooth_objective, z, method='L-BFGS-B',
                           jac=smooth_gradient,
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-14})
            return res.x
        except Exception:
            return z

    def get_initial_guesses():
        guesses = []
        side = 5
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65, 1.5, 0.6, 1.3, 0.75]
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
                r[i] = min(w0, h0) / side * 0.45
            guesses.append((x, y, r, w0))

        # Random starts
        for seed_val in range(10):
            rng = np.random.default_rng(seed_val * 7919 + 13)
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.03, 0.09)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0))

        return guesses

    guesses = get_initial_guesses()

    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 40:
            break

        h0 = 2.0 - w0

        # Step 1: Force relax
        x, y, r = force_relax(x0, y0, r0, w0, h0, max_iter=1500)

        # Step 2: Expand radii
        x, y, r = expand_radii(x, y, r, w0, h0, time_budget=3.0)

        # Step 3: Local search
        x, y, r = local_search(x, y, r, w0, h0, time_budget=6.0)

        # Step 4: Expand again
        x, y, r = expand_radii(x, y, r, w0, h0, time_budget=3.0)

        # Check validity
        if is_valid_fast(x, y, r, w0, h0):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w0
                best_result = z.copy()

        # Step 5: SLSQP polish if time permits
        if time.time() < deadline - 25 and best_result is not None:
            z = np.empty(3 * N + 1)
            z[0::3][:N] = x
            z[1::3][:N] = y
            z[2::3][:N] = r
            z[-1] = w0

            budget = min(10.0, deadline - time.time() - 20)
            if budget > 3:
                z_refined = slsqp_polish(z, time_budget=budget)
                if z_refined is not None:
                    xr = z_refined[0::3][:N]
                    yr = z_refined[1::3][:N]
                    rr = z_refined[2::3][:N]
                    wr = z_refined[-1]
                    hr = 2.0 - wr
                    if is_valid_fast(xr, yr, rr, wr, hr):
                        val_r = float(np.sum(rr))
                        if val_r > best_value:
                            best_value = val_r
                            best_result = z_refined.copy()

        # Step 6: L-BFGS-B
        if time.time() < deadline - 18:
            z = np.empty(3 * N + 1)
            z[0::3][:N] = x
            z[1::3][:N] = y
            z[2::3][:N] = r
            z[-1] = w0

            budget = min(8.0, deadline - time.time() - 15)
            if budget > 2:
                z_lbfgs = lbfgsb_optimize(z, time_budget=budget)
                xl = z_lbfgs[0::3][:N]
                yl = z_lbfgs[1::3][:N]
                rl = z_lbfgs[2::3][:N]
                wl = z_lbfgs[-1]
                hl = 2.0 - wl
                if is_valid_fast(xl, yl, rl, wl, hl):
                    val_l = float(np.sum(rl))
                    if val_l > best_value:
                        best_value = val_l
                        best_result = z_lbfgs.copy()

    # Phase 2: Polish best result
    if best_result is not None and time.time() < deadline - 15:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Expand
        x, y, r = expand_radii(x, y, r, w, h, time_budget=min(4.0, deadline - time.time() - 12))
        if is_valid_fast(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                best_result = z.copy()

        # Local search
        if time.time() < deadline - 10:
            x, y, r = local_search(x, y, r, w, h, time_budget=min(8.0, deadline - time.time() - 7))
            if is_valid_fast(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = x
                    z[1::3][:N] = y
                    z[2::3][:N] = r
                    z[-1] = w
                    best_result = z.copy()

        # SLSQP on best
        if time.time() < deadline - 8:
            z = np.empty(3 * N + 1)
            z[0::3][:N] = x
            z[1::3][:N] = y
            z[2::3][:N] = r
            z[-1] = w
            budget = min(6.0, deadline - time.time() - 5)
            if budget > 2:
                z_refined = slsqp_polish(z, time_budget=budget)
                if z_refined is not None:
                    xr = z_refined[0::3][:N]
                    yr = z_refined[1::3][:N]
                    rr = z_refined[2::3][:N]
                    wr = z_refined[-1]
                    hr = 2.0 - wr
                    if is_valid_fast(xr, yr, rr, wr, hr):
                        val_r = float(np.sum(rr))
                        if val_r > best_value:
                            best_value = val_r
                            best_result = z_refined.copy()

        # Final expand
        if time.time() < deadline - 3:
            x, y, r = expand_radii(x, y, r, w, h, time_budget=min(2.0, deadline - time.time() - 1))
            if is_valid_fast(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = x
                    z[1::3][:N] = y
                    z[2::3][:N] = r
                    z[-1] = w
                    best_result = z.copy()

    # Phase 3: Try different aspect ratios
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w_orig = best_result[-1]
        h_orig = 2.0 - w_orig

        for w_try in [w_orig * 0.88, w_orig * 1.12, w_orig * 0.82, w_orig * 1.18]:
            if time.time() > deadline - 5:
                break
            if w_try < 0.3 or w_try > 1.7:
                continue
            h_try = 2.0 - w_try

            sx = w_try / w_orig
            sy = h_try / h_orig
            xs = x * sx
            ys = y * sy
            rs = r * min(sx, sy)

            xs, ys, rs = force_relax(xs, ys, rs, w_try, h_try, max_iter=800)
            xs, ys, rs = expand_radii(xs, ys, rs, w_try, h_try, time_budget=2.0)
            xs, ys, rs = local_search(xs, ys, rs, w_try, h_try, time_budget=3.0)
            xs, ys, rs = expand_radii(xs, ys, rs, w_try, h_try, time_budget=1.5)

            if is_valid_fast(xs, ys, rs, w_try, h_try):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = xs
                    z[1::3][:N] = ys
                    z[2::3][:N] = rs
                    z[-1] = w_try
                    best_result = z.copy()

    if best_result is None:
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.01
        return np.column_stack((x, y, r))

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Final safety shrink
    for scale in [1.0, 0.99999, 0.9999, 0.999]:
        r_trial = r * scale
        if is_valid_fast(x, y, r_trial, best_result[-1], 2.0 - best_result[-1], tol=1e-9):
            r = r_trial
            break

    return np.column_stack((x, y, r))
