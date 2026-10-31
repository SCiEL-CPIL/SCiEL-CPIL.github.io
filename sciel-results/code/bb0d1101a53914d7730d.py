"""Circle packing in perimeter-4 rectangle: smooth SLSQP + vectorized polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE_MARGIN = 15.0  # seconds before the 150s kill


def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN

    best_value = 0.0
    best_result = None

    def smooth_obj(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.25 or h < 0.25:
            return 1e6

        obj = -np.sum(r)

        # Smooth overlap penalty using log-sum-exp
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                req = r[i] + r[j]
                # Smooth: use softplus-like
                if d < req * 1.5:
                    overlap = req - d
                    obj += 500.0 * overlap * overlap

        # Smooth boundary penalty
        for i in range(N):
            # Use smooth hinge: max(0, a)^2 approximated
            for val, lim in [(x[i], r[i]), (w - x[i], r[i]), (y[i], r[i]), (h - y[i], r[i])]:
                if val < r[i] * 1.5:
                    pen = r[i] - val
                    if pen > 0:
                        obj += 500.0 * pen * pen

        # Aspect ratio soft constraint
        if w < 0.5:
            obj += 500.0 * (0.5 - w) ** 2
        elif w > 1.5:
            obj += 500.0 * (w - 1.5) ** 2

        return obj

    def check_validity(z, tol=1e-7):
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
        side = 5  # 5x5 grid for 21 circles (4 rows of 5 + 1)

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = []
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(12):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = []
            for i in range(N):
                cx = rng.uniform(0.1, w0 - 0.1)
                cy = rng.uniform(0.1, h0 - 0.1)
                r0 = rng.uniform(0.05, 0.12)
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 3:
            break

        z0_relaxed = _force_relax(z0, max_iter=300)

        maxiter = int(min(400, (deadline - time.time()) * 25))
        if maxiter < 50:
            break
        try:
            res = minimize(smooth_obj, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt
            elif not valid:
                # Try repair: shrink radii
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                for shrink in [0.95, 0.90, 0.85]:
                    z_repair[r_idx] = z_opt[r_idx] * shrink
                    valid, val = check_validity(z_repair)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_repair
                        break

        except Exception:
            continue

    # Phase 2: Polish
    if best_result is not None and time.time() < deadline - 5:
        polished = _polish_result(best_result, deadline - (time.time() - t0))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    if best_result is None:
        # Fallback
        side = 5
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    r = r * (1 - 1e-6)
    return np.column_stack((x, y, r))


def _force_relax(z, max_iter=300):
    """Vectorized force-directed relaxation."""
    z = z.copy().astype(float)
    n = N
    dt = 0.004

    for _ in range(max_iter):
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 25
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 25
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 25
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 25

        # Pairwise repulsion (vectorized)
        for i in range(n):
            dx = x[i] - x
            dy = y[i] - y
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r
            mask = (d < min_d) & (d > 1e-12) & (np.arange(n) != i)
            if np.any(mask):
                overlap = min_d[mask] - d[mask]
                fx[i] += np.sum((dx[mask] / d[mask]) * overlap * 12)
                fy[i] += np.sum((dy[mask] / d[mask]) * overlap * 12)
                for j in np.where(mask)[0]:
                    fx[j] -= (dx[mask] / d[mask]) * overlap * 12
                    fy[j] -= (dy[mask] / d[mask]) * overlap * 12

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish with vectorized force relaxation + aggressive radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.003
    max_iter = int(min(1500, time_budget * 80))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 35
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 35
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 35
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 35

        for i in range(n):
            dx = x[i] - x
            dy = y[i] - y
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r
            mask = (d < min_d) & (d > 1e-12) & (np.arange(n) != i)
            if np.any(mask):
                overlap = min_d[mask] - d[mask]
                fx[i] += np.sum((dx[mask] / d[mask]) * overlap * 18)
                fy[i] += np.sum((dy[mask] / d[mask]) * overlap * 18)
                for j in np.where(mask)[0]:
                    fx[j] -= (dx[mask] / d[mask]) * overlap * 18
                    fy[j] -= (dy[mask] / d[mask]) * overlap * 18

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Aggressive uniform radius growth via binary search
    for _ in range(50):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.5
        best_scale = 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                best_scale = mid
                lo = mid
            else:
                hi = mid
        if best_scale > 1.0001:
            r = r * best_scale
        else:
            break

    # Non-uniform: try growing each circle individually
    for i in range(n):
        if time.time() > time_budget:
            break
        lo, hi = r[i], r[i] * 1.3
        for _ in range(20):
            mid = (lo + hi) / 2
            r_trial = r.copy()
            r_trial[i] = mid
            if _is_valid(x, y, r_trial, w, h):
                r[i] = mid
                lo = mid
            else:
                hi = mid

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-7):
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
