"""Circle packing in perimeter-4 rectangle: LM (least_squares) + expansion polish."""
import numpy as np
import time
from scipy.optimize import least_squares, minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x - r < -tol) or np.any(x + r > w + tol):
            return False, 0.0
        if np.any(y - r < -tol) or np.any(y + r > h + tol):
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def _residuals(z):
        """Residuals = signed gap violations (positive means overlap).
        z = [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1, w]
        """
        x = z[0:N]
        y = z[N:2 * N]
        r = z[2 * N:3 * N]
        w = z[3 * N]
        h = 2.0 - w

        res = []
        # Circle-circle gaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                res.append(r[i] + r[j] - d)  # positive = overlap
        # Wall gaps (positive = outside)
        for i in range(N):
            res.append(r[i] - x[i])
            res.append(x[i] + r[i] - w)
            res.append(r[i] - y[i])
            res.append(y[i] + r[i] - h)
        return np.array(res)

    def _lm_optimize(x0, y0, r0, w0, time_budget):
        """Levenberg-Marquardt optimization of gap violations."""
        z0 = np.concatenate([x0, y0, r0, [w0]])
        maxiter = int(min(2000, max(200, (time_budget - (time.time() - start_time)) * 100)))

        try:
            res = least_squares(
                _residuals, z0,
                method='trf',  # Trust region reflective, supports bounds
                bounds=([0.0] * (3 * N) + [0.3],
                        [2.0] * (3 * N) + [1.7]),
                max_nfev=maxiter,
                xtol=1e-14, ftol=1e-14, gtol=1e-14,
                verbose=0
            )
            z_opt = res.x
            x = z_opt[0:N]
            y = z_opt[N:2 * N]
            r = z_opt[2 * N:3 * N]
            w = z_opt[3 * N]
            h = 2.0 - w
            valid, val = check_validity(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
        except Exception:
            pass
        return None

    def _expand_radii(x, y, r, w, h, time_budget):
        """Iteratively expand each circle to max valid radius."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for i in range(N):
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
                if r_max > r[i] + 1e-8:
                    lo, hi = r[i], r_max
                    for _ in range(20):
                        mid = (lo + hi) / 2
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            if dx * dx + dy * dy < (mid + r[j]) ** 2 - 1e-10:
                                valid = False
                                break
                        if valid and mid <= r_wall + 1e-10:
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-9:
                        r[i] = lo
                        improved = True
        return r

    def _force_relax(x, y, r, w, h, max_iter=1000):
        """Force-directed relaxation to resolve overlaps."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.003
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            fx -= np.maximum(r - x, 0) * 80
            fx += np.maximum(x + r - w, 0) * 80
            fy -= np.maximum(r - y, 0) * 80
            fy += np.maximum(y + r - h, 0) * 80
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        max_overlap = max(max_overlap, overlap)
                        if d > 1e-12:
                            ux = dx / d
                            uy = dy / d
                        else:
                            ux, uy = 1.0, 0.0
                        f = overlap * 20.0
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f
            if max_overlap < 1e-10:
                break
            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
        return x, y

    def _slsqp_refine(x0, y0, r0, w0, time_budget):
        """SLSQP with smooth sqrt penalty for final refinement."""
        def objective(z):
            x = z[0:N]
            y = z[N:2 * N]
            r = z[2 * N:3 * N]
            w = z[3 * N]
            h = 2.0 - w
            if w < 0.2 or h < 0.2 or np.any(r < 0):
                return 1e6
            obj = -np.sum(r)
            pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    ov = max(0.0, r[i] + r[j] - d)
                    pen += np.sqrt(ov + 1e-12)
            obj += 500 * pen
            bp = 0.0
            for i in range(N):
                bp += np.sqrt(max(0.0, r[i] - x[i]) + 1e-12)
                bp += np.sqrt(max(0.0, x[i] + r[i] - w) + 1e-12)
                bp += np.sqrt(max(0.0, r[i] - y[i]) + 1e-12)
                bp += np.sqrt(max(0.0, y[i] + r[i] - h) + 1e-12)
            obj += 500 * bp
            return obj

        z0 = np.concatenate([x0, y0, r0, [w0]])
        bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
        maxiter = int(min(300, max(50, (time_budget - (time.time() - start_time)) * 20)))
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 40})
            z_opt = res.x
            x = z_opt[0:N]
            y = z_opt[N:2 * N]
            r = z_opt[2 * N:3 * N]
            w = z_opt[3 * N]
            h = 2.0 - w
            valid, val = check_validity(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
        except Exception:
            pass
        return None

    # --- Generate diverse initial guesses ---
    def get_initial_guesses():
        guesses = []
        rng = np.random.default_rng(123)

        # Grid-based with various aspect ratios
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55, 1.1, 0.9]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            # Hexagonal-ish packing
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            # Try to fit circles in a hex pattern
            spacing = min(w0, h0) / 5.0
            row = 0
            col = 0
            idx = 0
            while idx < N and row < 10:
                for c in range(6):
                    if idx >= N:
                        break
                    xx = (c + 0.5) * spacing + (0.5 * spacing if row % 2 else 0)
                    yy = (row + 0.5) * spacing * 0.866
                    if xx + spacing / 2 <= w0 and yy + spacing / 2 <= h0:
                        x[idx] = xx
                        y[idx] = yy
                        r[idx] = spacing * 0.45
                        idx += 1
                row += 1
            # Fill any remaining
            for idx in range(N):
                if r[idx] == 0:
                    x[idx] = rng.uniform(0.1, w0 - 0.1)
                    y[idx] = rng.uniform(0.1, h0 - 0.1)
                    r[idx] = 0.05
            guesses.append((x, y, r, w0, h0))

        # Random starts
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r[i] = rng.uniform(0.04, 0.12)
                x[i] = rng.uniform(r[i], w0 - r[i])
                y[i] = rng.uniform(r[i], h0 - r[i])
            guesses.append((x, y, r, w0, h0))

        # Square-ish with jitter
        for _ in range(5):
            w0 = 1.0
            h0 = 1.0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, 5)
                x[i] = (col + 0.5) / 5.0 + rng.uniform(-0.03, 0.03)
                y[i] = (row + 0.5) / 5.0 + rng.uniform(-0.03, 0.03)
                r[i] = 0.09
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 40:
            break

        # Force relax to resolve overlaps
        x_rel, y_rel = _force_relax(x0, y0, r0, w0, h0, max_iter=500)

        # LM optimization (the core structural change)
        remaining = deadline - (time.time() - start_time) - 20
        if remaining < 10:
            break
        result = _lm_optimize(x_rel, y_rel, r0.copy(), w0, deadline - 15)

        if result is not None:
            x, y, r, w, h, val = result
            # Expand radii after LM
            r = _expand_radii(x, y, r, w, h, deadline - 10)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)

        # Also try SLSQP from the same start as backup
        if time.time() < deadline - 25:
            result2 = _slsqp_refine(x_rel, y_rel, r0.copy(), w0, deadline - 12)
            if result2 is not None:
                x2, y2, r2, w2, h2, val2 = result2
                r2 = _expand_radii(x2, y2, r2, w2, h2, deadline - 8)
                valid2, val2 = check_validity(x2, y2, r2, w2, h2)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = (x2.copy(), y2.copy(), r2.copy(), w2, h2)

    # --- Polish best result ---
    if best_result is not None and time.time() < deadline - 25:
        x, y, r, w, h = best_result

        # LM polish
        result = _lm_optimize(x, y, r, w, deadline - 15)
        if result is not None:
            x, y, r, w, h, val = result
            r = _expand_radii(x, y, r, w, h, deadline - 10)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)

        # SLSQP polish
        if time.time() < deadline - 12:
            result = _slsqp_refine(x, y, r, w, deadline - 8)
            if result is not None:
                x2, y2, r2, w2, h2, val2 = result
                r2 = _expand_radii(x2, y2, r2, w2, h2, deadline - 5)
                valid2, val2 = check_validity(x2, y2, r2, w2, h2)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = (x2.copy(), y2.copy(), r2.copy(), w2, h2)

        # Final expansion
        if best_result is not None and time.time() < deadline - 3:
            x, y, r, w, h = best_result
            r = _expand_radii(x, y, r, w, h, deadline - 1)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
